"""
Change3D Base Implementation
극단적으로 최소화된 Video Modeling for Change Detection
이것보다 더 줄이면 Video Modeling 패러다임이 무너짐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Change3D_v2Base(nn.Module):
    """
    최소 Change3D 네트워크
    
    핵심 아이디어만:
    1. Perception Frame (학습 가능한 "관찰자")
    2. Pretrained 2D Backbone (Spatial 특징 추출)
    3. 3D Conv 1개 (Time 차원만 섞기)
    4. Perception Feature 추출
    
    A2Net Base와 비슷한 수준의 단순함!
    """
    
    def __init__(self, num_classes=1):
        super(Change3D_v2Base, self).__init__()
        
        # 🎯 핵심 1: Perception Frame
        # bi-temporal 이미지 사이에 끼워넣을 "관찰자"
        self.perception_frame = nn.Parameter(
            torch.randn(1, 3, 256, 256) * 0.02
        )
        
        # 🎯 핵심 2: 2D Pretrained Backbone
        # Spatial 특징은 이미 학습된 ResNet이 추출!
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  # 64 channels
            backbone.layer2   # 128 channels, H/8
        )
        
        # 🎯 핵심 3: 3D Conv (Time 차원만 섞기)
        # [I1, P, I2] 사이의 시간적 관계만 학습
        self.conv3d = nn.Conv3d(
            128, 256,
            kernel_size=(3, 3, 3),  # Time도 섞음
            stride=(1, 2, 2),       # Spatial만 stride
            padding=(1, 1, 1)
        )
        self.bn3d = nn.BatchNorm3d(256)
        
        # 🎯 핵심 4: 변화 탐지 헤드
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, t1, t2):
        """
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        Returns:
            change_map: [B, num_classes, H, W]
        """
        B, C, H, W = t1.shape
        
        # 1. Perception Frame 확장
        P = self.perception_frame.expand(B, -1, -1, -1)  # [B, 3, H, W]
        
        # 2. [I1, P, I2] 각각 2D Backbone으로 특징 추출
        # 3개 프레임을 배치로 합침
        frames = torch.stack([t1, P, t2], dim=1)  # [B, 3, 3, H, W]
        frames = frames.view(B * 3, C, H, W)      # [B*3, 3, H, W]
        
        # 2D Backbone으로 Spatial 특징 추출 (Pretrained!)
        feat = self.encoder(frames)  # [B*3, 128, H/8, W/8]
        
        # 3. 3D로 reshape
        _, C_feat, H_feat, W_feat = feat.shape
        feat = feat.view(B, 3, C_feat, H_feat, W_feat)  # [B, 3, 128, H/8, W/8]
        
        # Time을 채널 차원 앞으로
        feat = feat.permute(0, 2, 1, 3, 4)  # [B, 128, 3, H/8, W/8]
        
        # 4. 3D Conv로 Time 차원 섞기
        # [I1, P, I2] 사이의 시간적 관계 학습!
        feat = F.relu(self.bn3d(self.conv3d(feat)))  # [B, 256, 3, H/16, W/16]
        
        # 5. Perception Feature 추출
        # Time 차원에서 중간(index=1) = Perception Frame의 특징
        perception_feat = feat[:, :, 1, :, :]  # [B, 256, H/16, W/16]
        
        # 6. 변화 맵 생성
        change = self.head(perception_feat)  # [B, 1, H/16, W/16]
        
        # 7. 원본 크기로 복원
        change_map = F.interpolate(
            change,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return change_map


if __name__ == "__main__":
    print("="*60)
    print("Change3D Base - Minimal Video Modeling for Change Detection")
    print("="*60)
    
    # 모델 생성
    model = Change3DBase(num_classes=1)
    
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
    
    # A2Net과 비교
    print(f"\nComparison with A2Net Base:")
    print(f"  A2Net Base: ~3.5M params")
    print(f"  Change3D Base: {total_params/1e6:.1f}M params")
    print(f"  Ratio: {total_params/3.5e6:.1f}x")
    
    print("\n" + "="*60)
    print("✓ Change3D Base implementation complete!")
    print("  - Only ~100 lines (vs 200+ before)")
    print("  - Pretrained ResNet18 (Spatial)")
    print("  - Just 1 Conv3D layer (Time only)")
    print("  - Similar complexity to A2Net Base")
    print("="*60)