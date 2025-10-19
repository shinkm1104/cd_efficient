"""
Change3D Base Implementation
극단적으로 최소화된 Video Modeling for Change Detection
이것보다 더 줄이면 Video Modeling 패러다임이 무너짐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Change3DBase(nn.Module):
    """
    최소 Change3D 네트워크
    
    핵심 아이디어만:
    1. Perception Frame (학습 가능한 "관찰자")
    2. 3D Convolution (비디오처럼 처리)
    3. Time 차원에서 Perception Feature 추출
    
    A2Net Base와 비슷한 수준의 단순함!
    """
    
    def __init__(self, num_classes=1):
        super(Change3DBase, self).__init__()
        
        # 🎯 핵심 1: Perception Frame
        # bi-temporal 이미지 사이에 끼워넣을 "관찰자"
        self.perception_frame = nn.Parameter(
            torch.randn(1, 3, 256, 256) * 0.02
        )
        
        # 🎯 핵심 2: 3D Convolution (비디오 인코더)
        # [I1, P, I2]를 Time 차원으로 처리
        # Time 차원도 같이 섞어야 변화 학습 가능!
        
        # Stage 1: [B, 3, 3, H, W] -> [B, 32, 3, H/2, W/2]
        self.conv3d_1 = nn.Conv3d(
            3, 32, 
            kernel_size=(3, 3, 3),  # Time 차원도 섞음!
            stride=(1, 2, 2), 
            padding=(1, 1, 1)       # Time padding 추가
        )
        self.bn1 = nn.BatchNorm3d(32)
        
        # Stage 2: [B, 32, 3, H/2, W/2] -> [B, 64, 3, H/4, W/4]
        self.conv3d_2 = nn.Conv3d(
            32, 64,
            kernel_size=(3, 3, 3),  # Time 차원 섞음
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)
        
        # Stage 3: [B, 64, 3, H/4, W/4] -> [B, 128, 3, H/8, W/8]
        self.conv3d_3 = nn.Conv3d(
            64, 128,
            kernel_size=(3, 3, 3),  # Time 차원 섞음
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(128)
        
        # Stage 4: [B, 128, 3, H/8, W/8] -> [B, 256, 3, H/16, W/16]
        self.conv3d_4 = nn.Conv3d(
            128, 256,
            kernel_size=(3, 3, 3),  # Time 차원 섞음
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.bn4 = nn.BatchNorm3d(256)
        
        # 🎯 핵심 3: 변화 탐지 헤드 (A2Net처럼 1x1 Conv)
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
        
        # 2. 비디오 구성: [I1, P, I2] - Time 차원으로 쌓기
        # 논문의 핵심 아이디어!
        video = torch.stack([t1, P, t2], dim=2)  # [B, 3, 3, H, W]
        #                                             ↑ Time=3
        
        # 3. 3D Convolution으로 특징 추출
        x = F.relu(self.bn1(self.conv3d_1(video)))  # [B, 32, 3, H/2, W/2]
        x = F.relu(self.bn2(self.conv3d_2(x)))      # [B, 64, 3, H/4, W/4]
        x = F.relu(self.bn3(self.conv3d_3(x)))      # [B, 128, 3, H/8, W/8]
        x = F.relu(self.bn4(self.conv3d_4(x)))      # [B, 256, 3, H/16, W/16]
        
        # 4. Perception Feature 추출
        # Time 차원에서 중간(index=1) = Perception Frame의 특징
        # 이게 Change3D의 핵심 트릭!
        perception_feat = x[:, :, 1, :, :]  # [B, 256, H/16, W/16]
        
        # 5. 변화 맵 생성 (A2Net처럼 1x1 Conv)
        change = self.head(perception_feat)  # [B, 1, H/16, W/16]
        
        # 6. 원본 크기로 복원
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
    print("  - Only ~130 lines (vs 200+ before)")
    print("  - No separate Backbone/Decoder classes")
    print("  - Just 4 Conv3D layers + 1 head")
    print("  - Similar complexity to A2Net Base")
    print("="*60)