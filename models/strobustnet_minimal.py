"""
STRobustNet Base Implementation
극단적으로 최소화된 Spatial-Temporal Robust Representation
이것보다 더 줄이면 Robust Representation 패러다임이 무너짐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class STRobustNetBase(nn.Module):
    """
    최소 STRobustNet 네트워크
    
    핵심 아이디어만:
    1. Learnable Embeddings (Universal Robust Representations)
    2. Simple Alignment (MCA로 현재 이미지에 맞춤)
    3. Classification (Robust Rep로 활성화)
    4. Change Detection (분류 결과 비교)
    """
    
    def __init__(self, num_classes=1, num_representations=4):
        super(STRobustNetBase, self).__init__()
        
        self.num_repr = num_representations
        self.feat_dim = 256  # 특징 차원
        
        # 🎯 핵심 1: Feature Extractor (ResNet-18 사용)
        # 논문은 ResNet-18 사용
        backbone = resnet18(pretrained=True)
        # Conv1~Layer3까지만 사용 (1/8 해상도)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3  # [B, 256, H/8, W/8]
        )
        
        # 🎯 핵심 2: Universal Robust Representations (학습 가능)
        # 전체 데이터셋에서 학습되는 보편적 표현
        self.learnable_embeddings = nn.Parameter(
            torch.randn(num_representations, self.feat_dim) * 0.02
        )
        
        # 🎯 핵심 3: Alignment Network (간단한 MCA)
        # Universal → Specific으로 정렬
        self.mca = MultiHeadCrossAttention(
            dim=self.feat_dim,
            num_heads=4
        )
        
        # Layer Norm & MLP
        self.norm1 = nn.LayerNorm(self.feat_dim)
        self.norm2 = nn.LayerNorm(self.feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim * 2, self.feat_dim)
        )
        
        # 🎯 핵심 4: Prediction Head (분류 → 변화)
        # 2 * num_repr → 2 (change/no-change)
        self.prediction_head = nn.Sequential(
            nn.Conv2d(2 * num_representations, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, t1, t2):
        """
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        Returns:
            change_map: [B, 1, H, W]
        """
        B, C, H, W = t1.shape
        
        # 1. 특징 추출
        feat1 = self.encoder(t1)  # [B, 256, H/8, W/8]
        feat2 = self.encoder(t2)  # [B, 256, H/8, W/8]
        
        _, C_feat, H_feat, W_feat = feat1.shape
        
        # 2. 특징을 시퀀스로 변환
        # [B, C, H, W] → [B, H*W, C]
        feat1_seq = feat1.flatten(2).permute(0, 2, 1)  # [B, N, C]
        feat2_seq = feat2.flatten(2).permute(0, 2, 1)  # [B, N, C]
        
        # Temporal context: 두 시점 특징 합치기
        feat_seq = torch.cat([feat1_seq, feat2_seq], dim=1)  # [B, 2N, C]
        
        # 3. Universal Robust Representations 가져오기
        # [num_repr, C] → [B, num_repr, C]
        embeddings = self.learnable_embeddings.unsqueeze(0).expand(B, -1, -1)
        
        # 4. Alignment: Universal → Specific
        # MCA로 현재 이미지 특징과 정렬
        aligned = self.mca(
            query=embeddings + self.learnable_embeddings.unsqueeze(0),
            key=feat_seq,
            value=feat_seq
        ) + embeddings
        
        aligned = self.norm1(aligned)
        
        # MLP
        aligned = self.mlp(self.norm2(aligned)) + aligned  # [B, num_repr, C]
        
        # 5. Classification Representations 생성
        # Robust Representations로 각 픽셀 활성화
        # [B, num_repr, C] @ [B, C, H*W] → [B, num_repr, H*W]
        
        # feat1을 [B, C, H*W]로
        feat1_flat = feat1.flatten(2)  # [B, C, H*W]
        feat2_flat = feat2.flatten(2)  # [B, C, H*W]
        
        # Classification: dot product
        # [B, num_repr, C] @ [B, C, H*W] → [B, num_repr, H*W]
        class_rep1 = torch.bmm(aligned, feat1_flat)  # [B, num_repr, H*W]
        class_rep2 = torch.bmm(aligned, feat2_flat)  # [B, num_repr, H*W]
        
        # Softmax normalization
        class_rep1 = F.softmax(class_rep1, dim=1)  # Class 차원으로 normalize
        class_rep2 = F.softmax(class_rep2, dim=1)
        
        # Reshape to spatial
        class_rep1 = class_rep1.view(B, self.num_repr, H_feat, W_feat)
        class_rep2 = class_rep2.view(B, self.num_repr, H_feat, W_feat)
        
        # 6. Concatenate & Predict
        # [B, 2*num_repr, H/8, W/8] → [B, 1, H/8, W/8]
        class_rep = torch.cat([class_rep1, class_rep2], dim=1)
        change = self.prediction_head(class_rep)  # [B, 1, H/8, W/8]
        
        # 7. Upsample to original size
        change_map = F.interpolate(
            change,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return change_map


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross Attention"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        """
        Args:
            query: [B, N_q, C]
            key: [B, N_k, C]
            value: [B, N_v, C]
        Returns:
            out: [B, N_q, C]
        """
        B, N_q, C = query.shape
        N_k = key.shape[1]
        
        # Project
        Q = self.q_proj(query)  # [B, N_q, C]
        K = self.k_proj(key)    # [B, N_k, C]
        V = self.v_proj(value)  # [B, N_k, C]
        
        # Reshape for multi-head
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_q, D]
        K = K.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_k, D]
        V = V.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_k, D]
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_k]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ V  # [B, H, N_q, D]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, N_q, C)
        
        # Output projection
        out = self.o_proj(out)
        
        return out


if __name__ == "__main__":
    print("="*60)
    print("STRobustNet Base - Spatial-Temporal Robust Representation")
    print("="*60)
    
    # 모델 생성
    model = STRobustNetBase(num_classes=1, num_representations=4)
    
    # 테스트 입력
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    
    # Forward
    output = model(t1, t2)
    
    print(f"\nInput shapes:")
    print(f"  t1: {t1.shape}")
    print(f"  t2: {t2.shape}")
    print(f"\nOutput shape:")
    print(f"  change_map: {output.shape}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "="*60)
    print("✓ STRobustNet Base implementation complete!")
    print("  - Learnable Embeddings (Universal Robust Rep)")
    print("  - Simple Alignment (1 MCA layer)")
    print("  - Classification Representations")
    print("  - Change Detection via Classification")
    print("="*60)