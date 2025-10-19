"""
모델별 하이퍼파라미터 설정
각 논문에서 제시한 최적 설정값
"""

def get_model_config(model_name):
    """
    모델별 최적 하이퍼파라미터 반환
    
    Args:
        model_name: 모델 이름 (MODEL_LIST에 있는 이름)
    
    Returns:
        dict: 모델 설정 딕셔너리
    """
    """
    1. weight_decay: 가중치 감쇠. 
    모델의 가중치가 너무 커지는 것을 막아 과적합(Overfitting)을 방지하는 정규화(Regularization) 값입니다.
    
    2. betas: Adam, AdamW 등 적응형 옵티마이저에서 사용하는 계수. 
    과거 그래디언트의 이동 평균을 계산할 때 사용되는 감쇠율 (β₁, β₂) 입니다.

    3.eps: Epsilon. 옵티마이저 계산 시 분모가 0이 되는 것을 방지하여 
    수치적 안정성을 확보하기 위해 더해주는 매우 작은 값 (e.g., 1e-8) 입니다.
    
    4. scheduler: 학습률 스케줄러. 학습 과정 중 학습률(learning rate)을 
    동적으로 조정하여 모델이 안정적으로 최적값에 수렴하도록 돕습니다.
    
    5.momentum: 모멘텀. 경사 하강법(Gradient Descent)에 관성을 더해주는 값. 
    이전 업데이트 방향을 유지하여 진동을 줄이고 수렴 속도를 높입니다.
    """
    # 
    
    configs = {
        ### base config
        'BaseCD': { 
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'scheduler': 'cosine',
            'momentum': None,  # SGD용
        },
        
        'A2Net': {
            'optimizer': 'adam',
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'betas': (0.9, 0.99),
            'eps': 1e-8,
            'scheduler': 'poly',
            'momentum': None,
        },
        
        'Changer': {
            'optimizer': 'adamw',
            'learning_rate': 0.001,  # MiT 백본시 0.0001
            'weight_decay': 0.05,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'scheduler': 'poly',
            'momentum': None,
        },
        
        'Change3D': {
            'optimizer': 'adam',
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'scheduler': 'poly',
            'momentum': None,
        },
        
        'USSFC-Net': {
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'weight_decay': 0.0005,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'scheduler': None,
            'momentum': None,
        },
        
        'EATDer': {
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01,  # 기본값 사용 (논문 미명시)
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'scheduler': None,
            'momentum': None,
        },
        
        'ChangeMamba': {
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 5e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'scheduler': None,
            'momentum': None,
        },
        
        'STRobustNet': {
            'optimizer': 'sgd',
            'learning_rate': 0.01,  # LEVIR/WHU용, SYSU는 0.005
            'weight_decay': 0.0005,
            'betas': None,
            'eps': None,
            'scheduler': 'poly',
            'momentum': 0.9,
        },
    }
    
    if model_name not in configs:
        print(f"Warning: {model_name} config not found, using default BaseCD config")
        return configs['BaseCD']
    
    return configs[model_name]


def adjust_config_for_dataset(model_name, dataset_name, config):
    """
    특정 데이터셋에 대한 모델 설정 조정
    
    Args:
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        config: 기본 모델 설정
    
    Returns:
        dict: 조정된 설정
    """
    # STRobustNet의 경우 SYSU 데이터셋에서 다른 학습률 사용
    if model_name == 'STRobustNet' and dataset_name == 'SYSU-CD':
        config = config.copy()
        config['learning_rate'] = 0.005
    
    # Changer의 경우 MiT 백본 사용시 학습률 조정
    # (백본 정보가 있을 때 추가)
    
    return config