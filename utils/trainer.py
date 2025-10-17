"""
Change Detection 학습 관리
"""

import time
import json
from pathlib import Path
from datetime import timedelta
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .metrics import CDMetrics


class CDTrainer:
    """Change Detection 학습 클래스"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Args:
            model: CD 모델
            optimizer: 옵티마이저
            criterion: 손실 함수
            device: 학습 장치
            checkpoint_dir: 체크포인트 저장 경로
            scheduler: 학습률 스케줄러
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        
        # 메트릭 기록
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_iou': [],
            'val_iou': [],
        }
        
        self.best_f1 = 0
        self.best_iou = 0
        self.current_epoch = 0
        self.current_iter = 0
    
    def train_epoch(self, loader) -> Tuple[float, Dict]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        metrics = CDMetrics()
        
        pbar = tqdm(
            loader,
            desc=f'[Train] Epoch {self.current_epoch}',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            leave=False  # 완료 후 프로그레스바 제거
        )
        
        for batch in pbar:
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            label = batch['label'].to(self.device)
            
            # Forward & Backward
            output = self.model(img1, img2)
            loss = self.criterion(output, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 메트릭 업데이트
            total_loss += loss.item()
            metrics.update(output.detach(), label)
            self.current_iter += 1
            
            # 프로그레스바 업데이트
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()  # 명시적으로 닫기
        avg_loss = total_loss / len(loader)
        results = metrics.get_metrics()
        
        return avg_loss, results
    
    def validate(self, loader, desc='Val') -> Tuple[float, Dict]:
        """검증 또는 테스트"""
        self.model.eval()
        total_loss = 0
        metrics = CDMetrics()
        
        pbar = tqdm(
            loader,
            desc=f'[{desc}]',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]',
            leave=False  # 완료 후 프로그레스바 제거
        )
        
        with torch.no_grad():
            for batch in pbar:
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                label = batch['label'].to(self.device)
                
                output = self.model(img1, img2)
                loss = self.criterion(output, label)
                
                total_loss += loss.item()
                metrics.update(output, label)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()  # 명시적으로 닫기
        avg_loss = total_loss / len(loader)
        results = metrics.get_metrics()
        
        return avg_loss, results
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'iterations': self.current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_f1': self.best_f1,
            'best_iou': self.best_iou,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 일반 체크포인트
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        
        # Best 모델
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved! F1: {metrics['f1']:.4f}, IoU: {metrics['iou']:.4f}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_f1 = checkpoint.get('best_f1', 0)
        self.best_iou = checkpoint.get('best_iou', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iterations', 0)
        
        return checkpoint
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        val_interval: int = 1,
        save_interval: int = 10
    ):
        """전체 학습 프로세스"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Learning rate 출력
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch}/{epochs} - LR: {current_lr:.6f}")
            else:
                print(f"\nEpoch {epoch}/{epochs}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['train_iou'].append(train_metrics['iou'])
            
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            # Validation
            if epoch % val_interval == 0:
                val_loss, val_metrics = self.validate(val_loader, 'Val')
                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_metrics['f1'])
                self.history['val_iou'].append(val_metrics['iou'])
                
                print(f"Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, "
                      f"IoU: {val_metrics['iou']:.4f}")
                
                # Best model 체크
                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.best_iou = val_metrics['iou']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # 주기적 저장
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, train_metrics)
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
            
            # 시간 정보
            epoch_time = time.time() - epoch_start
            eta = (time.time() - start_time) / epoch * (epochs - epoch)
            print(f"Time: {epoch_time:.2f}s, ETA: {str(timedelta(seconds=int(eta)))}")
        
        # 학습 완료
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training Completed!")
        print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"Best F1: {self.best_f1:.4f}, Best IoU: {self.best_iou:.4f}")
        print("="*60)
    
    def test(self, test_loader) -> Dict:
        """테스트 평가"""
        print("\n" + "="*60)
        print("Testing")
        print("="*60)
        
        # Best model 로드
        best_path = self.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            checkpoint = self.load_checkpoint(best_path)
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Test
        start_time = time.time()
        test_loss, test_metrics = self.validate(test_loader, 'Test')
        test_time = time.time() - start_time
        
        print(f"\nTest Results:")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  IoU: {test_metrics['iou']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  OA: {test_metrics['oa']:.4f}")
        print(f"  Kappa: {test_metrics['kappa']:.4f}")
        print(f"  Time: {test_time:.2f}s")
        
        return test_metrics
    
    def measure_inference_speed(self, loader, num_batches: int = 10) -> Dict:
        """추론 속도 측정"""
        self.model.eval()
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                
                # Warmup
                if i == 0:
                    _ = self.model(img1, img2)
                    continue
                
                # 시간 측정
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = self.model(img1, img2)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times.append(time.time() - start)
        
        avg_time = np.mean(times)
        batch_size = loader.batch_size
        fps = batch_size / avg_time
        
        print(f"\nInference Speed:")
        print(f"  Batch time: {avg_time*1000:.2f}ms")
        print(f"  FPS: {fps:.2f} images/sec")
        print(f"  Single image: {avg_time*1000/batch_size:.2f}ms")
        
        return {
            'batch_time_ms': avg_time * 1000,
            'fps': fps,
            'single_image_ms': avg_time * 1000 / batch_size
        }
    
    def save_results(self, save_dir: Path, model_name: str, dataset_name: str):
        """결과 저장"""
        results = {
            'model': model_name,
            'dataset': dataset_name,
            'best_f1': self.best_f1,
            'best_iou': self.best_iou,
            'history': self.history,
            'total_iterations': self.current_iter,
        }
        
        save_path = save_dir / 'results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {save_path}")