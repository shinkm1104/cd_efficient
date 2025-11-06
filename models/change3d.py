"""
Change3D with X3D Backbone
X3D 백본을 사용한 강력한 Video Modeling for Change Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

# X3D 백본 import
from models.x3d import create_x3d


class Change3DBase(nn.Module):
    """
    Change3D Base Class
    
    핵심 아이디어:
    1. Perception Frame (학습 가능한 "관찰자")
    2. X3D Backbone (강력한 비디오 특징 추출)
    3. Time 차원에서 Perception Feature 추출
    """
    
    def __init__(self, num_classes=1, width_factor=2.0, depth_factor=5.0, 
                 x3d_out_dim=192, version='l'):
        super(Change3DBase, self).__init__()
        
        self.version = version
        
        # 🎯 핵심 1: Perception Frame
        self.perception_frame = nn.Parameter(
            torch.randn(1, 3, 256, 256) * 0.02
        )
        
        # 🎯 핵심 2: X3D Backbone 설정
        self.x3d = create_x3d(
            input_channel=3,
            input_clip_length=3,  # [I1, P, I2] = 3 frames
            input_crop_size=256,
            model_num_class=400,  # Kinetics-400 (가중치 호환성)
            width_factor=width_factor,
            depth_factor=depth_factor,
            dropout_rate=0.5,
            head_output_with_global_average=False
        )
        
        # 🔥 사전학습 가중치 불러오기
        self.load_pretrained_weights(version)
        
        # X3D의 head 제거 (백본만 사용)
        self.x3d.blocks = self.x3d.blocks[:-1]  # classification head 제거
        
        # 🎯 핵심 3: 변화 탐지 헤드
        # Time 차원 처리를 위한 Conv3D
        self.temporal_conv = nn.Conv3d(
            x3d_out_dim, 256,
            kernel_size=(3, 1, 1),  # Time축만 처리
            padding=(1, 0, 0)
        )
        
        # 최종 변화 맵 생성 헤드
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def load_pretrained_weights(self, version='l'):
        """사전학습된 X3D 가중치 불러오기"""
        # 가중치 파일 경로
        weight_path = f'./pretrained/Change3D/X3D_{version.upper()}.pyth'
        
        if not os.path.exists(weight_path):
            print(f"⚠️ Warning: Pretrained weights not found at {weight_path}")
            print("   Using random initialization")
            print(f"   Download weights from: https://dl.fbaipublicfiles.com/pytorchvideo/x3d/x3d_{version}.pyth")
            return
        
        try:
            print(f"Loading pretrained weights from {weight_path}...")
            
            # 가중치 불러오기
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # checkpoint 구조 확인
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 현재 모델의 state_dict
            model_dict = self.x3d.state_dict()
            
            # Head를 제외한 가중치만 필터링
            pretrained_dict = {}
            for k, v in state_dict.items():
                # head/classifier 관련 레이어 제외
                if any(skip in k for skip in ['head', 'proj', 'fc', 'classifier']):
                    continue
                
                # 키 이름 매칭 시도
                clean_key = k.replace('module.', '')  # DDP 래핑 제거
                
                if clean_key in model_dict:
                    if model_dict[clean_key].shape == v.shape:
                        pretrained_dict[clean_key] = v
                    else:
                        print(f"   Shape mismatch for {clean_key}: {v.shape} vs {model_dict[clean_key].shape}")
                elif 'blocks.' in clean_key and clean_key in model_dict:
                    # blocks.0.xxx 형태 처리
                    if model_dict[clean_key].shape == v.shape:
                        pretrained_dict[clean_key] = v
            
            # 가중치 업데이트
            model_dict.update(pretrained_dict)
            self.x3d.load_state_dict(model_dict, strict=False)
            
            print(f"✅ Successfully loaded X3D-{version.upper()} pretrained weights")
            print(f"   Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
            
            # 로드된 주요 블록 확인
            loaded_blocks = set()
            for k in pretrained_dict.keys():
                if 'blocks.' in k:
                    block_num = k.split('.')[1]
                    loaded_blocks.add(block_num)
            if loaded_blocks:
                print(f"   Loaded blocks: {sorted(loaded_blocks)}")
            
        except Exception as e:
            print(f"❌ Error loading pretrained weights: {e}")
            print("   Using random initialization")
    
    def forward(self, t1, t2):
        """
        Args:
            t1: Time 1 이미지 [B, 3, H, W]
            t2: Time 2 이미지 [B, 3, H, W]
        Returns:
            change_map: [B, num_classes, H, W]
        """
        B, C, H, W = t1.shape
        
        # 1. Perception Frame 확장
        P = self.perception_frame.expand(B, -1, -1, -1)
        
        # 2. 비디오 구성: [I1, P, I2]
        # X3D는 [B, C, T, H, W] 형식 필요
        video = torch.stack([t1, P, t2], dim=2)  # [B, 3, 3, H, W]
        
        # 3. X3D Backbone으로 특징 추출
        x = video
        for i, block in enumerate(self.x3d.blocks):
            x = block(x)
        # x: [B, C_out, T', H', W'] (공간/시간 차원 축소됨)
        
        # 4. Temporal Conv로 Perception Feature 강조
        x = self.temporal_conv(x)  # [B, 256, T', H', W']
        
        # 5. Perception Feature 추출 (중간 time frame)
        # Perception Frame의 특징을 추출
        t_mid = x.size(2) // 2
        perception_feat = x[:, :, t_mid, :, :]  # [B, 256, H', W']
        
        # 6. 변화 맵 생성
        change = self.head(perception_feat)  # [B, num_classes, H', W']
        
        # 7. 원본 크기로 복원
        change_map = F.interpolate(
            change, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        return change_map


class Change3DXS(Change3DBase):
    """
    Change3D with X3D-XS (eXtra Small)
    - Width Factor: 0.5
    - Depth Factor: 2.2
    - 가장 작고 빠른 버전
    """
    def __init__(self, num_classes=1):
        super().__init__(
            num_classes=num_classes,
            width_factor=0.5,
            depth_factor=2.2,
            x3d_out_dim=48,  # 0.5 * 96
            version='xs'
        )


class Change3DS(Change3DBase):
    """
    Change3D with X3D-S (Small)
    - Width Factor: 1.0
    - Depth Factor: 1.0
    - 작고 효율적인 버전
    """
    def __init__(self, num_classes=1):
        super().__init__(
            num_classes=num_classes,
            width_factor=1.0,
            depth_factor=1.0,
            x3d_out_dim=96,
            version='s'
        )


class Change3DM(Change3DBase):
    """
    Change3D with X3D-M (Medium)
    - Width Factor: 1.5
    - Depth Factor: 2.9
    - 균형잡힌 중간 버전
    """
    def __init__(self, num_classes=1):
        super().__init__(
            num_classes=num_classes,
            width_factor=1.5,
            depth_factor=2.9,
            x3d_out_dim=144,  # 1.5 * 96
            version='m'
        )


class Change3DL(Change3DBase):
    """
    Change3D with X3D-L (Large)
    - Width Factor: 2.0
    - Depth Factor: 5.0
    - 가장 크고 정확한 버전 (기본값)
    """
    def __init__(self, num_classes=1):
        super().__init__(
            num_classes=num_classes,
            width_factor=2.0,
            depth_factor=5.0,
            x3d_out_dim=192,
            version='l'
        )


# 기본 Change3D 클래스 (하위 호환성)
class Change3D(Change3DL):
    """기본 Change3D = Change3DL (Large 버전)"""
    pass


if __name__ == "__main__":
    print("="*60)
    print("Change3D with X3D Backbone (Multiple Versions)")
    print("="*60)
    
    # 각 버전별 테스트
    versions = [
        ('XS', Change3DXS),
        ('S', Change3DS),
        ('M', Change3DM),
        ('L', Change3DL)
    ]
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Version':<10} {'Parameters':<15} {'Size (MB)':<12}")
    print("-"*60)
    
    for name, ModelClass in versions:
        model = ModelClass(num_classes=1)
        model.eval()
        
        # 파라미터 계산
        total_params = sum(p.numel() for p in model.parameters())
        param_size = total_params * 4 / 1024 / 1024  # FP32
        
        print(f"{name:<10} {total_params/1e6:>10.2f}M     {param_size:>8.1f} MB")
    
    print("="*60)
    
    # X3D-L로 forward pass 테스트
    print("\nTesting Change3DL (default)...")
    model = Change3DL(num_classes=1)
    model.eval()
    
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(t1, t2)
    
    print(f"Input: {t1.shape}")
    print(f"Output: {output.shape}")
    print("\n✅ All Change3D versions ready!")