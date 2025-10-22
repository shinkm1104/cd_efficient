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


class Change3DX3D(nn.Module):
    """
    X3D 백본을 사용한 Change3D 네트워크
    
    핵심 아이디어:
    1. Perception Frame (학습 가능한 "관찰자")
    2. X3D Backbone (강력한 비디오 특징 추출)
    3. Time 차원에서 Perception Feature 추출
    """
    
    def __init__(self, num_classes=1, x3d_version='l', pretrained=True):
        super(Change3DX3D, self).__init__()
        
        # 🎯 핵심 1: Perception Frame
        self.perception_frame = nn.Parameter(
            torch.randn(1, 3, 256, 256) * 0.02
        )
        
        # 🎯 핵심 2: X3D Backbone 설정
        if x3d_version.lower() == 'l':
            # X3D-L 설정 (Large 버전)
            self.x3d = create_x3d(
                input_channel=3,
                input_clip_length=3,  # [I1, P, I2] = 3 frames
                input_crop_size=256,
                model_num_class=400,  # Kinetics-400 (가중치 호환성)
                width_factor=2.0,
                depth_factor=5.0,  # X3D-L의 핵심: depth가 5배
                dropout_rate=0.5,
                head_output_with_global_average=False
            )
            x3d_out_dim = 192  # X3D-L의 출력 차원
            
        elif x3d_version.lower() == 'xs':
            # X3D-XS 설정 (eXtra Small 버전)
            self.x3d = create_x3d(
                input_channel=3,
                input_clip_length=3,
                input_crop_size=256,
                model_num_class=400,
                width_factor=2.0,
                depth_factor=2.2,  # X3D-XS는 2.2배
                dropout_rate=0.5,
                head_output_with_global_average=False
            )
            x3d_out_dim = 192  # X3D-XS의 출력 차원
        else:
            raise ValueError(f"Unknown X3D version: {x3d_version}")
        
        # 🔥 사전학습 가중치 불러오기
        if pretrained:
            self.load_pretrained_weights(x3d_version)
        
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


# 기본 Change3D 클래스 (간단한 인터페이스)
class Change3D(nn.Module):
    """기본 Change3D - X3D-L 사용"""
    
    def __init__(self, num_classes=1):
        super(Change3D, self).__init__()
        # X3D-L을 기본으로 사용 (가중치 있음)
        self.model = Change3DX3D(
            num_classes=num_classes,
            x3d_version='l',  # Large 버전
            pretrained=True   # 사전학습 가중치 사용
        )
    
    def forward(self, t1, t2):
        return self.model(t1, t2)


if __name__ == "__main__":
    print("="*60)
    print("Change3D with X3D-L Backbone (Pretrained)")
    print("="*60)
    
    # X3D-L 버전 테스트
    print("\nInitializing Change3D with X3D-L...")
    model = Change3DX3D(num_classes=1, x3d_version='l', pretrained=True)
    
    # 모델을 evaluation 모드로
    model.eval()
    
    # 테스트 입력
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    
    print(f"\nInput shapes:")
    print(f"  t1: {t1.shape}")
    print(f"  t2: {t2.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(t1, t2)
    
    print(f"\nOutput shape:")
    print(f"  change_map: {output.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params/1e6:.1f}M")
    print(f"  Trainable: {trainable_params/1e6:.1f}M")
    
    # 메모리 사용량 (추정)
    param_size = total_params * 4 / 1024 / 1024  # FP32 기준
    print(f"  Estimated size: {param_size:.1f} MB (FP32)")
    
    print("\n✅ Change3D with X3D-L ready for training!")
    print("   - Perception Frame: Learning temporal changes")
    print("   - X3D-L Backbone: Powerful spatiotemporal features")
    print("   - Pretrained: Kinetics-400 initialization")