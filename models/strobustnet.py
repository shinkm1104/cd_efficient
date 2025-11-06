"""
STRobustNet Base - True Minimal
- Learnable Embeddings 제거! (핵심 아이디어 제거)
- 그냥 백본 + Difference만!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class STRobustNetSkeleton(nn.Module):
    """
    최소 Change Detection 네트워크
    
    필수 구성요소만:
    1. 특징 추출 (ResNet-18)
    2. 차이 계산 (절댓값)
    3. 변화 맵 생성 (1x1 conv)
    
    STRobustNet 특징 없음!
    """
    
    def __init__(self, num_classes=1):
        super(STRobustNetSkeleton, self).__init__()
        
        # 특징 추출기 (ResNet-18 Layer3까지)
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )  # [B, 256, H/8, W/8]
        
        # 변화 탐지 헤드 (차이를 변화맵으로 변환)
        self.head = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, t1, t2):
        """
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        Returns:
            change_map: [B, num_classes, H, W]
        """
        # 1. 특징 추출
        feat1 = self.encoder(t1)  # [B, 256, H/8, W/8]
        feat2 = self.encoder(t2)  # [B, 256, H/8, W/8]
        
        # 2. 차이 계산 (가장 단순한 방법)
        diff = torch.abs(feat1 - feat2)  # [B, 256, H/8, W/8]
        
        # 3. 변화 맵 생성
        change = self.head(diff)  # [B, 1, H/8, W/8]
        
        # 4. 원본 크기로 복원
        h, w = t1.shape[-2:]
        change_map = F.interpolate(
            change, size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        
        return change_map


if __name__ == "__main__":
    print("="*60)
    print("STRobustNet Base - True Minimal")
    print("="*60)
    
    model = STRobustNetSkeleton(num_classes=1)
    
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    
    output = model(t1, t2)
    
    print(f"\nInput shapes:")
    print(f"  t1: {t1.shape}")
    print(f"  t2: {t2.shape}")
    print(f"\nOutput shape:")
    print(f"  change_map: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "="*60)
    print("✓ STRobustNet Base (True Minimal) complete!")
    print("  - ResNet-18 Siamese Encoder")
    print("  - Difference Fusion (256 channels)")
    print("  - 1x1 Conv Head")
    print("  - No Learnable Embeddings!")
    print("  - No Classification!")
    print("  - ~30 lines only!")
    print("="*60)