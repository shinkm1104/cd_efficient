"""
Change Detection 설정 관리
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    root_dir: str = "dataset"
    dataset_names: List[str] = field(default_factory=lambda: [
        'LEVIR-CD+', 'WHU-CD', 'CLCD', 
        'CaBuAr-CD', 'S2Looking-CD', 'SEN1Floods11-CD'
    ])
    img_size: int = 256
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    model_name: str = "minimal_cd"
    num_classes: int = 1  # Binary CD
    pretrained: bool = True
    backbone: str = "mobilenet_v2"


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    
    # Loss
    loss_fn: str = "bce_dice"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'bce': 0.5,
        'dice': 0.5
    })
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.0001
    
    # Gradient clipping
    clip_grad: bool = True
    max_grad_norm: float = 1.0
    
    # Validation
    val_interval: int = 1
    
    # Checkpoint
    save_best: bool = True
    save_interval: int = 10


@dataclass 
class ExpConfig:
    """실험 관련 설정"""
    exp_name: str = "minimal_cd_exp"
    exp_dir: str = "experiments"
    seed: int = 42
    device: str = "cuda"
    
    # Logging
    log_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "change-detection"
    
    # Debug
    debug: bool = False
    num_samples: Optional[int] = None  # Debug시 샘플 수 제한


@dataclass
class Config:
    """전체 설정 클래스"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
    
    def __post_init__(self):
        """설정 후처리"""
        # 실험 디렉토리 생성
        self.exp_path = Path(self.exp.exp_dir) / self.exp.exp_name
        self.exp_path.mkdir(parents=True, exist_ok=True)
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = self.exp_path / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 로그 디렉토리
        self.log_dir = self.exp_path / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 결과 디렉토리
        self.result_dir = self.exp_path / "results"
        self.result_dir.mkdir(exist_ok=True)
    
    def save(self, path: Optional[str] = None):
        """설정 저장"""
        if path is None:
            path = self.exp_path / "config.yaml"
        
        config_dict = asdict(self)
        # Path 객체를 문자열로 변환
        for key in ['exp_path', 'checkpoint_dir', 'log_dir', 'result_dir']:
            if key in config_dict:
                config_dict[key] = str(config_dict[key])
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """설정 로드"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 중첩된 dataclass 처리
        config = cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            train=TrainConfig(**config_dict.get('train', {})),
            exp=ExpConfig(**config_dict.get('exp', {}))
        )
        
        return config
    
    def print_config(self):
        """설정 출력"""
        print("\n" + "="*50)
        print("Configuration")
        print("="*50)
        
        for section_name in ['data', 'model', 'train', 'exp']:
            section = getattr(self, section_name)
            print(f"\n[{section_name.upper()}]")
            for key, value in asdict(section).items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*50)


def get_config(
    config_file: Optional[str] = None,
    **kwargs
) -> Config:
    """
    설정 생성 또는 로드
    
    Args:
        config_file: 설정 파일 경로 (있으면 로드)
        **kwargs: 오버라이드할 설정값
    
    Returns:
        Config 객체
    """
    if config_file and os.path.exists(config_file):
        config = Config.load(config_file)
        print(f"Config loaded from {config_file}")
    else:
        config = Config()
        print("Using default config")
    
    # kwargs로 설정 오버라이드
    for key, value in kwargs.items():
        if '.' in key:
            # nested key (e.g., 'train.lr')
            section, param = key.split('.', 1)
            if hasattr(config, section):
                setattr(getattr(config, section), param, value)
        else:
            # top-level key
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


# 사전 정의된 설정 템플릿
PRESET_CONFIGS = {
    'debug': {
        'exp.debug': True,
        'exp.num_samples': 100,
        'train.epochs': 2,
        'data.batch_size': 4,
        'exp.log_interval': 1,
    },
    'fast': {
        'train.epochs': 30,
        'train.val_interval': 5,
        'train.save_interval': 10,
    },
    'full': {
        'train.epochs': 200,
        'train.patience': 30,
        'data.augment': True,
        'train.early_stopping': True,
    }
}


def get_preset_config(preset: str = 'default', **kwargs) -> Config:
    """
    사전 정의된 설정 템플릿 사용
    
    Args:
        preset: 'debug', 'fast', 'full', 'default'
        **kwargs: 추가 오버라이드 설정
    
    Returns:
        Config 객체
    """
    config = Config()
    
    if preset in PRESET_CONFIGS:
        preset_kwargs = PRESET_CONFIGS[preset]
        for key, value in preset_kwargs.items():
            section, param = key.split('.')
            setattr(getattr(config, section), param, value)
    
    # 추가 kwargs 적용
    for key, value in kwargs.items():
        if '.' in key:
            section, param = key.split('.', 1)
            setattr(getattr(config, section), param, value)
    
    return config


if __name__ == "__main__":
    # 테스트 코드
    
    # 기본 설정
    config = Config()
    config.print_config()
    
    # 설정 저장
    config.save("test_config.yaml")
    
    # 설정 로드
    loaded_config = Config.load("test_config.yaml")
    
    # Preset 사용
    debug_config = get_preset_config('debug')
    print("\nDebug Config:")
    debug_config.print_config()