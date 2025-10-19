"""
USSFC-Net Base Model (Extreme Minimal)
- Backbone: ResNet-18 (Pretrained, Pseudo-Siamese / Non-weight-sharing)
- Feature Fusion: Difference (256 channels)
- Stage: 1개만! (Layer3까지)
- No Multi-scale, No MSDConv, No SSFC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class USSFCNetBase(nn.Module):
    """
    USSFC-Net Base Network (극단적 최소)
    
    핵심:
    1. Pseudo-Siamese Encoder (Non-weight-sharing, 2개 Encoder) - Layer3까지만!
    2. Difference Fusion (256 channels)
    3. Simple Head
    
    Multi-scale 없음! Stage 1개만!
    """
    
    def __init__(self, num_classes=1):
        super(USSFCNetBase, self).__init__()
        
        # 🎯 핵심 1: Pseudo-Siamese Encoder (Non-weight-sharing)
        # Time 1용 Encoder (Layer3까지만!)
        backbone1 = resnet18(pretrained=True)
        self.encoder1 = nn.Sequential(
            backbone1.conv1,
            backbone1.bn1,
            backbone1.relu,
            backbone1.maxpool,
            backbone1.layer1,
            backbone1.layer2,
            backbone1.layer3
        )  # [B, 256, H/8, W/8]
        
        # Time 2용 Encoder (다른 가중치!)
        backbone2 = resnet18(pretrained=True)
        self.encoder2 = nn.Sequential(
            backbone2.conv1,
            backbone2.bn1,
            backbone2.relu,
            backbone2.maxpool,
            backbone2.layer1,
            backbone2.layer2,
            backbone2.layer3
        )  # [B, 256, H/8, W/8]
        
        # 🎯 핵심 2: Simple Head
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
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
        
        # 1. Pseudo-Siamese Feature Extraction (각자 다른 Encoder!)
        feat1 = self.encoder1(t1)  # [B, 256, H/8, W/8]
        feat2 = self.encoder2(t2)  # [B, 256, H/8, W/8]
        
        # 2. Feature Difference
        diff = torch.abs(feat1 - feat2)  # [B, 256, H/8, W/8]
        
        # 3. Change Map
        change = self.head(diff)  # [B, 1, H/8, W/8]
        
        # 4. Upsample
        change_map = F.interpolate(
            change,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return change_map


if __name__ == "__main__":
    print("="*60)
    print("USSFC-Net Base - Extreme Minimal (1 Stage Only)")
    print("="*60)
    
    model = USSFCNetBase(num_classes=1)
    
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    
    output = model(t1, t2)
    
    print(f"\nInput shapes:")
    print(f"  t1: {t1.shape}")
    print(f"  t2: {t2.shape}")
    print(f"\nOutput shape:")
    print(f"  change_map: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "="*60)
    print("✓ USSFC-Net Base implementation complete!")
    print("  - Pseudo-Siamese ResNet-18 (Non-weight-sharing, 2 encoders)")
    print("  - Layer3까지만 (1 Stage)")
    print("  - Difference Fusion (256 channels)")
    print("  - Simple Head")
    print("  - No Multi-scale!")
    print("  - No MSDConv!")
    print("  - No SSFC!")
    print("  - ~50 lines only!")
    print("="*60)