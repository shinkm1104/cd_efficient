"""
A2Net Minimal Implementation
A2Net에서 효율성 모듈(NAM, PCIM, SAM)을 제거한 최소 구현
논문: Lightweight Remote Sensing Change Detection (Li et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class A2NetMinimal(nn.Module):
    """
    A2Net 최소 구현
    
    원본 A2Net 구조:
    1. MobileNetV2 백본 ✓ (구현)
    2. NAM (Neighbor Aggregation) ✗ (제거)
    3. Feature Difference ✓ (구현)
    4. PCIM (Progressive Change Identifying) ✗ (제거)
    5. SAM (Supervised Attention) ✗ (제거)
    6. 디코더 ✓ (단순화하여 구현)
    """
    
    def __init__(self, num_classes=1):
        super(A2NetMinimal, self).__init__()
        
        # ========== 1. 백본 네트워크 ==========
        # MobileNetV2를 불러와서 특징 추출에 사용
        # pretrained=True: ImageNet으로 사전학습된 가중치 사용
        mobilenet = mobilenet_v2(pretrained=True)
        
        # MobileNetV2의 각 스테이지를 분리
        # 논문에서는 5개 스테이지를 사용, 우리는 마지막 4개만 사용
        self.stage1 = mobilenet.features[0:2]   # 해상도 1/2, 채널 16
        self.stage2 = mobilenet.features[2:4]   # 해상도 1/4, 채널 24  
        self.stage3 = mobilenet.features[4:7]   # 해상도 1/8, 채널 32
        self.stage4 = mobilenet.features[7:14]  # 해상도 1/16, 채널 96
        self.stage5 = mobilenet.features[14:18] # 해상도 1/32, 채널 320
        
        # ========== 2. 채널 조정 레이어 ==========
        # MobileNet 출력 채널을 통일된 크기로 조정
        # 1x1 컨볼루션으로 채널 수만 변경 (공간 크기는 유지)
        self.adjust2 = nn.Conv2d(24, 64, kernel_size=1)   # 24ch → 64ch
        self.adjust3 = nn.Conv2d(32, 128, kernel_size=1)  # 32ch → 128ch
        self.adjust4 = nn.Conv2d(96, 256, kernel_size=1)  # 96ch → 256ch
        self.adjust5 = nn.Conv2d(320, 512, kernel_size=1) # 320ch → 512ch
        
        # ========== 3. 디코더 ==========
        # 점진적으로 해상도를 높이며 변화 정보 추출
        # 각 단계마다 3x3 컨볼루션 사용
        self.decoder5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder4 = nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1)  # concat 후 512ch
        self.decoder3 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)   # concat 후 256ch
        self.decoder2 = nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1)     # concat 후 128ch
        
        # ========== 4. 최종 분류 레이어 ==========
        # 1x1 컨볼루션으로 최종 변화 맵 생성
        # num_classes=1: 이진 변화 탐지 (변화/무변화)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward_single(self, x):
        """
        단일 시점 이미지의 다중 스케일 특징 추출
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
        
        Returns:
            f2, f3, f4, f5: 각 스테이지의 특징
        """
        # 순차적으로 각 스테이지 통과
        f1 = self.stage1(x)      # [B, 16, H/2, W/2]
        f2 = self.stage2(f1)     # [B, 24, H/4, W/4]
        f3 = self.stage3(f2)     # [B, 32, H/8, W/8]
        f4 = self.stage4(f3)     # [B, 96, H/16, W/16]
        f5 = self.stage5(f4)     # [B, 320, H/32, W/32]
        
        # 채널 수 조정
        f2 = self.adjust2(f2)    # [B, 64, H/4, W/4]
        f3 = self.adjust3(f3)    # [B, 128, H/8, W/8]
        f4 = self.adjust4(f4)    # [B, 256, H/16, W/16]
        f5 = self.adjust5(f5)    # [B, 512, H/32, W/32]
        
        return f2, f3, f4, f5
    
    def forward(self, t1, t2):
        """
        변화 탐지 수행
        
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        
        Returns:
            change_map: 변화 맵 [B, num_classes, H, W]
        """
        # ========== 1. 특징 추출 ==========
        # 각 시점 이미지에서 다중 스케일 특징 추출
        f1_2, f1_3, f1_4, f1_5 = self.forward_single(t1)
        f2_2, f2_3, f2_4, f2_5 = self.forward_single(t2)
        
        # ========== 2. 특징 차분 (Feature Difference) ==========
        # 두 시점 특징의 절댓값 차이 계산
        # A2Net 논문 Eq. (2) 구현
        diff2 = torch.abs(f1_2 - f2_2)  # [B, 64, H/4, W/4]
        diff3 = torch.abs(f1_3 - f2_3)  # [B, 128, H/8, W/8]
        diff4 = torch.abs(f1_4 - f2_4)  # [B, 256, H/16, W/16]
        diff5 = torch.abs(f1_5 - f2_5)  # [B, 512, H/32, W/32]
        
        # ========== 3. 디코더 (Bottom-up) ==========
        # Stage 5 (가장 깊은 레벨)
        x = self.decoder5(diff5)  # [B, 256, H/32, W/32]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 업샘플링 후: [B, 256, H/16, W/16]
        
        # Stage 4 + Skip connection
        x = torch.cat([x, diff4], dim=1)  # [B, 512, H/16, W/16]
        x = self.decoder4(x)  # [B, 128, H/16, W/16]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 업샘플링 후: [B, 128, H/8, W/8]
        
        # Stage 3 + Skip connection
        x = torch.cat([x, diff3], dim=1)  # [B, 256, H/8, W/8]
        x = self.decoder3(x)  # [B, 64, H/8, W/8]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 업샘플링 후: [B, 64, H/4, W/4]
        
        # Stage 2 + Skip connection
        x = torch.cat([x, diff2], dim=1)  # [B, 128, H/4, W/4]
        x = self.decoder2(x)  # [B, 32, H/4, W/4]
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        # 업샘플링 후: [B, 32, H, W] (원본 크기로 복원)
        
        # ========== 4. 최종 출력 ==========
        change_map = self.final(x)  # [B, 1, H, W]
        
        return change_map


if __name__ == "__main__":
    # ========== 모델 테스트 ==========
    model = A2NetMinimal(num_classes=1)
    
    # 더미 입력 생성
    batch_size = 2
    t1 = torch.randn(batch_size, 3, 256, 256)
    t2 = torch.randn(batch_size, 3, 256, 256)
    
    # Forward pass
    output = model(t1, t2)
    print(f"Input shape: {t1.shape}")
    print(f"Output shape: {output.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")