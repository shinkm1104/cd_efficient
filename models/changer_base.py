"""
Changer Base Model (Extreme Minimal)
- Backbone: ResNet-18 (Pretrained, Weight-sharing Siamese)
- Feature Fusion: Concat (512 channels)
- Stage: 1개만! (Layer3까지)
- No Multi-scale, No Feature Exchange, No FDAF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ChangerBase(nn.Module):
    """
    Changer Base Network (극단적 최소)
    
    핵심:
    1. ResNet-18 Siamese Encoder (Weight-sharing) - Layer3까지만!
    2. Concat Fusion (512 channels)
    3. Simple Head
    
    Multi-scale 없음! Stage 1개만!
    """
    
    def __init__(self, num_classes=1):
        super(ChangerBase, self).__init__()
        
        # 🎯 핵심 1: Siamese Encoder (Weight-sharing)
        # Layer3까지만 (1개 Stage!)
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )  # [B, 256, H/8, W/8]
        
        # 🎯 핵심 2: Fusion Head (Concat이라 512 채널!)
        self.head = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1),  # 512 → 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
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
        
        # 1. Siamese Feature Extraction (가중치 공유!)
        feat1 = self.encoder(t1)  # [B, 256, H/8, W/8]
        feat2 = self.encoder(t2)  # [B, 256, H/8, W/8]
        
        # 2. Bi-temporal Fusion (Concat!)
        fused = torch.cat([feat1, feat2], dim=1)  # [B, 512, H/8, W/8]
        #                                              ↑ 2배!
        
        # 3. Change Map
        change = self.head(fused)  # [B, 1, H/8, W/8]
        
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
    print("Changer Base - Extreme Minimal (1 Stage Only)")
    print("="*60)
    
    model = ChangerBase(num_classes=1)
    
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
    print("✓ Changer Base implementation complete!")
    print("  - ResNet-18 Siamese Encoder (Weight-sharing)")
    print("  - Layer3까지만 (1 Stage)")
    print("  - Concat Fusion (512 channels)")
    print("  - Simple Head")
    print("  - No Multi-scale!")
    print("  - No Feature Exchange!")
    print("  - No FDAF!")
    print("  - ~50 lines only!")
    print("="*60)