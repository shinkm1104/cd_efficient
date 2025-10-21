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
    def __init__(self, num_classes=1):
        super(A2NetBase, self).__init__()
        
        # 단 1개 Stage! (Change3D처럼)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # [B, 256, H/4, W/4]
        
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, t1, t2):
        B, C, H, W = t1.shape
        
        feat1 = self.encoder(t1)
        feat2 = self.encoder(t2)
        
        diff = torch.abs(feat1 - feat2)
        
        change = self.head(diff)
        
        change_map = F.interpolate(
            change, size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        return change_map

