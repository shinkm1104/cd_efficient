"""
Change Detection Visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch


def prepare_image_for_visualization(img_tensor):
    """텐서를 시각화용 numpy 배열로 변환"""
    img = img_tensor.cpu().numpy().squeeze()
    
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # ImageNet 정규화 해제
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    
    return np.clip(img, 0, 1)


def create_comparison_overlay(gt_mask, pred_mask):
    """GT와 Prediction 비교 오버레이 생성"""
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3))
    
    # True Positive - 초록
    tp_mask = (gt_mask == 1) & (pred_mask == 1)
    overlay[tp_mask] = [0, 1, 0]
    
    # False Positive - 빨강
    fp_mask = (gt_mask == 0) & (pred_mask == 1)
    overlay[fp_mask] = [1, 0, 0]
    
    # False Negative - 노랑
    fn_mask = (gt_mask == 1) & (pred_mask == 0)
    overlay[fn_mask] = [1, 1, 0]
    
    # True Negative - 검정
    tn_mask = (gt_mask == 0) & (pred_mask == 0)
    overlay[tn_mask] = [0, 0, 0]
    
    return overlay


def overlay_comparison_on_image(image, gt_mask, pred_mask, alpha=0.7):
    """이미지 위에 비교 오버레이 합성"""
    overlay = create_comparison_overlay(gt_mask, pred_mask)
    change_mask = (gt_mask == 1) | (pred_mask == 1)
    
    result = image.copy()
    for c in range(3):
        result[:, :, c] = np.where(
            change_mask,
            image[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha,
            image[:, :, c]
        )
    
    return result


def save_visualization(pre_img, post_img, gt_mask, pred_mask, save_path, file_name):
    """Change Detection 결과 시각화 저장"""
    # 이미지 준비
    pre_img_vis = prepare_image_for_visualization(pre_img)
    post_img_vis = prepare_image_for_visualization(post_img)
    
    # 마스크 변환
    gt_mask_np = gt_mask.detach().cpu().numpy().squeeze()
    pred_mask_np = pred_mask.detach().cpu().numpy().squeeze()
    
    gt_mask_np = (gt_mask_np > 0.5).astype(np.uint8)
    pred_mask_np = (pred_mask_np > 0.5).astype(np.uint8)
    
    # Figure 생성 - 2x3 레이아웃
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Change Detection Results - {file_name}', fontsize=16)
    
    # 첫 번째 행
    axes[0, 0].imshow(pre_img_vis)
    axes[0, 0].set_title('T1 (Pre-change) Image')
    axes[0, 0].axis('off')
    
    pre_overlay = overlay_comparison_on_image(pre_img_vis, gt_mask_np, pred_mask_np, alpha=0.7)
    axes[0, 1].imshow(pre_overlay)
    axes[0, 1].set_title('T1 + Comparison Overlay')
    axes[0, 1].axis('off')
    
    comparison_overlay = create_comparison_overlay(gt_mask_np, pred_mask_np)
    axes[0, 2].imshow(comparison_overlay)
    axes[0, 2].set_title('Comparison Overlay')
    axes[0, 2].axis('off')
    
    # 두 번째 행
    axes[1, 0].imshow(post_img_vis)
    axes[1, 0].set_title('T2 (Post-change) Image')
    axes[1, 0].axis('off')
    
    post_overlay = overlay_comparison_on_image(post_img_vis, gt_mask_np, pred_mask_np, alpha=0.7)
    axes[1, 1].imshow(post_overlay)
    axes[1, 1].set_title('T2 + Comparison Overlay')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_mask_np, cmap='gray')
    axes[1, 2].set_title('Predicted Mask')
    axes[1, 2].axis('off')
    
    # 범례
    legend_elements = [
        Patch(facecolor='lime', label='True Positive'),
        Patch(facecolor='yellow', label='False Negative (Missed)'),
        Patch(facecolor='red', label='False Positive (Wrong)'),
        Patch(facecolor='black', label='True Negative')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()