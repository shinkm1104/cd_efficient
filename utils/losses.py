"""
Change Detection 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Binary Cross Entropy + Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        """
        Args:
            bce_weight: BCE loss 가중치
            dice_weight: Dice loss 가중치  
            smooth: Dice loss smoothing factor
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: 예측값 [B, 1, H, W]
            target: 실제값 [B, 1, H, W]
        Returns:
            Combined loss
        """
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        # Combined
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def get_loss_fn(loss_name='bce_dice', **kwargs):
    """손실 함수 생성 헬퍼"""
    if loss_name == 'bce_dice':
        return BCEDiceLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


if __name__ == "__main__":
    # 테스트
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    loss_fn = BCEDiceLoss()
    loss = loss_fn(pred, target)
    print(f"BCEDice Loss: {loss.item():.4f}")