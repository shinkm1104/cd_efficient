"""
Change Detection 평가 메트릭
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class CDMetrics:
    """Change Detection 메트릭 계산 클래스"""
    
    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: 클래스 수 (2 for binary CD)
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.total_samples = 0
    
    def update(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        threshold: float = 0.5
    ):
        """
        메트릭 업데이트
        
        Args:
            pred: 예측값 [B, 1, H, W] or [B, H, W]
            target: 실제값 [B, 1, H, W] or [B, H, W]
            threshold: 이진화 임계값
        """
        # Sigmoid 적용 (logits인 경우)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # 이진화
        pred_binary = (pred > threshold).float()
        
        # Flatten
        pred_binary = pred_binary.view(-1)
        target = target.view(-1)
        
        # 메트릭 계산
        self.tp += ((pred_binary == 1) & (target == 1)).sum().item()
        self.fp += ((pred_binary == 1) & (target == 0)).sum().item()
        self.tn += ((pred_binary == 0) & (target == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (target == 1)).sum().item()
        self.total_samples += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        모든 메트릭 계산 및 반환
        
        Returns:
            Dict containing:
                - precision
                - recall
                - f1
                - iou
                - oa (overall accuracy)
                - kappa
        """
        eps = 1e-7  # 0 나눗셈 방지
        
        # Basic metrics
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        # IoU (Intersection over Union)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        
        # Overall Accuracy
        total = self.tp + self.fp + self.tn + self.fn
        oa = (self.tp + self.tn) / (total + eps)
        
        # Kappa coefficient
        pe = ((self.tn + self.fn) * (self.tn + self.fp) + 
              (self.fp + self.tp) * (self.fn + self.tp)) / ((total + eps) ** 2)
        kappa = (oa - pe) / (1 - pe + eps)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'oa': oa,
            'kappa': kappa
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """혼동 행렬 반환"""
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])
    
    def __str__(self) -> str:
        """메트릭 출력"""
        metrics = self.get_metrics()
        return (
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"IoU: {metrics['iou']:.4f}, "
            f"OA: {metrics['oa']:.4f}, "
            f"Kappa: {metrics['kappa']:.4f}"
        )


class EarlyStopping:
    """Early Stopping 클래스"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'max'
    ):
        """
        Args:
            patience: 개선 없이 대기할 에폭 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.better = lambda x, y: x < y - min_delta
        else:
            self.better = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Early stopping 체크
        
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
        elif self.better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def calculate_metrics_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    배치 단위 메트릭 계산 (빠른 계산용)
    
    Args:
        pred: 예측값 [B, 1, H, W]
        target: 실제값 [B, 1, H, W]
        threshold: 이진화 임계값
    
    Returns:
        메트릭 딕셔너리
    """
    # Sigmoid 적용
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # 이진화
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # 메트릭 계산
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    
    eps = 1e-7
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }


if __name__ == "__main__":
    # 테스트 코드
    metrics = CDMetrics()
    
    # 더미 데이터
    pred = torch.rand(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # 메트릭 업데이트
    metrics.update(pred, target)
    
    # 결과 출력
    print(metrics)
    
    # 혼동 행렬
    print("\nConfusion Matrix:")
    print(metrics.get_confusion_matrix())