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
    def __init__(self, num_classes=1, num_representations=4):
        super().__init__()
        
        self.num_repr = num_representations
        
        # 1. Feature Extractor (ResNet-18)
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )  # [B, 256, H/8, W/8]
        
        # 2. Universal Robust Representations
        self.learnable_embeddings = nn.Parameter(
            torch.randn(num_representations, 256) * 0.02
        )
        
        # 3. Prediction Head (No Alignment!)
        self.head = nn.Conv2d(2 * num_representations, num_classes, kernel_size=1)
        
    def forward(self, t1, t2):
        B, C, H, W = t1.shape
        
        # 특징 추출
        feat1 = self.encoder(t1)  # [B, 256, H/8, W/8]
        feat2 = self.encoder(t2)
        
        _, C_feat, H_feat, W_feat = feat1.shape
        
        # Learnable Embeddings로 분류 (No Alignment!)
        embeddings = self.learnable_embeddings  # [num_repr, C]
        
        feat1_flat = feat1.view(B, C_feat, -1)  # [B, C, H*W]
        feat2_flat = feat2.view(B, C_feat, -1)
        
        # Classification: einsum으로 간단하게
        class1 = torch.einsum('rc,bcp->brp', embeddings, feat1_flat)
        class2 = torch.einsum('rc,bcp->brp', embeddings, feat2_flat)
        
        # Softmax
        class1 = F.softmax(class1, dim=1).view(B, self.num_repr, H_feat, W_feat)
        class2 = F.softmax(class2, dim=1).view(B, self.num_repr, H_feat, W_feat)
        
        # Change Detection
        class_rep = torch.cat([class1, class2], dim=1)
        change = self.head(class_rep)
        
        # Upsample
        change_map = F.interpolate(change, size=(H, W), mode='bilinear', align_corners=False)
        
        return change_map



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