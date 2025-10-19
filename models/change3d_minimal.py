"""
Change3D Base Implementation
Video Modeling 관점에서 Change Detection을 재해석
극단적으로 최소화된 3D Conv 구조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Simple3DBackbone(nn.Module):
    """
    X3D-L 스타일의 최소 3D 백본
    
    논문의 X3D-L을 극단적으로 단순화:
    - 4개 stage만 사용
    - Expansion, Squeeze-Excitation 제거
    - 기본 3D Conv만 사용
    """
    def __init__(self):
        super(Simple3DBackbone, self).__init__()
        
        # Stem: 초기 특징 추출
        self.stem = Conv3DBlock(3, 24, kernel_size=3, stride=(1,2,2), padding=1)
        
        # Stage 1: [B, 24, T, H/2, W/2] -> [B, 48, T, H/4, W/4]
        self.stage1 = nn.Sequential(
            Conv3DBlock(24, 48, kernel_size=3, stride=(1,2,2), padding=1),
            Conv3DBlock(48, 48, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 2: [B, 48, T, H/4, W/4] -> [B, 96, T, H/8, W/8]
        self.stage2 = nn.Sequential(
            Conv3DBlock(48, 96, kernel_size=3, stride=(1,2,2), padding=1),
            Conv3DBlock(96, 96, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 3: [B, 96, T, H/8, W/8] -> [B, 192, T, H/16, W/16]
        self.stage3 = nn.Sequential(
            Conv3DBlock(96, 192, kernel_size=3, stride=(1,2,2), padding=1),
            Conv3DBlock(192, 192, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 4: [B, 192, T, H/16, W/16] -> [B, 384, T, H/32, W/32]
        self.stage4 = nn.Sequential(
            Conv3DBlock(192, 384, kernel_size=3, stride=(1,2,2), padding=1),
            Conv3DBlock(384, 384, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - Time 차원 포함
        Returns:
            features: 멀티스케일 특징 리스트
        """
        x = self.stem(x)      # [B, 24, T, H/2, W/2]
        f1 = self.stage1(x)   # [B, 48, T, H/4, W/4]
        f2 = self.stage2(f1)  # [B, 96, T, H/8, W/8]
        f3 = self.stage3(f2)  # [B, 192, T, H/16, W/16]
        f4 = self.stage4(f3)  # [B, 384, T, H/32, W/32]
        
        return [f1, f2, f3, f4]


class SimpleDecoder(nn.Module):
    """
    최소 Decoder
    
    논문의 복잡한 feature aggregation 제거
    단순 Upsample만 사용
    """
    def __init__(self, in_channels=384, num_classes=1):
        super(SimpleDecoder, self).__init__()
        
        # 점진적 Upsample
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up5 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 최종 분류 레이어
        self.classifier = nn.Conv2d(24, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H/32, W/32]
        Returns:
            out: [B, num_classes, H, W]
        """
        x = self.up1(x)  # H/16
        x = self.up2(x)  # H/8
        x = self.up3(x)  # H/4
        x = self.up4(x)  # H/2
        x = self.up5(x)  # H
        x = self.classifier(x)
        return x


class Change3DBase(nn.Module):
    """
    최소 Change3D 네트워크
    
    핵심 아이디어:
    1. Perception Frame을 bi-temporal 이미지 사이에 삽입
    2. [I1, P, I2]를 비디오처럼 3D Conv로 처리
    3. Perception Feature만 추출하여 변화 예측
    
    이것보다 더 줄이면 Video Modeling 패러다임이 무너짐
    """
    
    def __init__(self, num_classes=1):
        super(Change3DBase, self).__init__()
        
        # Perception Frame: 학습 가능한 "관찰자" 프레임
        # 논문의 핵심 아이디어!
        self.perception_frame = nn.Parameter(
            torch.randn(1, 3, 256, 256) * 0.01
        )
        
        # 3D Backbone: 비디오 인코더
        self.backbone = Simple3DBackbone()
        
        # Decoder: 변화 맵 생성
        self.decoder = SimpleDecoder(in_channels=384, num_classes=num_classes)
        
    def forward(self, t1, t2):
        """
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        Returns:
            change_map: [B, num_classes, H, W]
        """
        B, C, H, W = t1.shape
        
        # 1. Perception Frame 확장 (배치 크기에 맞춤)
        P = self.perception_frame.expand(B, -1, -1, -1)  # [B, 3, H, W]
        
        # 2. 비디오 구성: [I1, P, I2]를 Time 차원으로 쌓기
        # [B, 3, H, W] x 3 -> [B, 3, 3, H, W]
        #                      ↑ Time=3
        video = torch.stack([t1, P, t2], dim=2)  # [B, C, T, H, W]
        
        # 3. 3D Backbone으로 특징 추출
        features = self.backbone(video)  # List of [B, C, T, H/s, W/s]
        
        # 4. Perception Feature 추출
        # Time 차원에서 중간(index=1) = Perception Frame의 특징
        f4 = features[-1]  # [B, 384, 3, H/32, W/32]
        perception_feat = f4[:, :, 1, :, :]  # [B, 384, H/32, W/32]
        
        # 5. 변화 맵 생성
        change_map = self.decoder(perception_feat)  # [B, 1, H, W]
        
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
    
    print("\n" + "="*60)
    print("✓ Change3D Base implementation complete!")
    print("="*60)