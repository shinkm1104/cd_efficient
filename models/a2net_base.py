"""
A2Net Base Implementation
극단적으로 최소화된 Change Detection 구조
이것보다 더 줄이면 CD가 불가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class A2NetBase(nn.Module):
    """
    최소 Change Detection 네트워크
    
    필수 구성요소만:
    1. 특징 추출 (백본)
    2. 차이 계산 (절댓값)
    3. 변화 맵 생성 (1x1 conv)
    """
    
    def __init__(self, num_classes=1):
        super(A2NetBase, self).__init__()
        
        # 특징 추출기 (MobileNetV2 전체)
        self.encoder = mobilenet_v2(pretrained=True).features
        # MobileNetV2 최종 출력: [B, 1280, H/32, W/32]
        
        # 변화 탐지 헤드 (차이를 변화맵으로 변환)
        self.head = nn.Conv2d(1280, num_classes, kernel_size=1)
        
    def forward(self, t1, t2):
        """
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        Returns:
            change_map: [B, num_classes, H, W]
        """
        # 1. 특징 추출
        feat1 = self.encoder(t1)  # [B, 1280, H/32, W/32]
        feat2 = self.encoder(t2)  # [B, 1280, H/32, W/32]
        
        # 2. 차이 계산 (가장 단순한 방법)
        diff = torch.abs(feat1 - feat2)  # [B, 1280, H/32, W/32]
        
        # 3. 변화 맵 생성
        change = self.head(diff)  # [B, 1, H/32, W/32]
        
        # 4. 원본 크기로 복원
        h, w = t1.shape[-2:]
        change_map = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        
        return change_map


if __name__ == "__main__":
    # 테스트
    model = A2NetBase()
    
    # 입력
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    
    # 출력
    output = model(t1, t2)
    print(f"Input: {t1.shape}")
    print(f"Output: {output.shape}")
    
    # 파라미터
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")