"""
Change Detection 데이터셋 클래스
n개 데이터셋 통합 처리
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional

# 지원 데이터셋 목록
SUPPORTED_DATASETS = [
        'LEVIR-CD+', 'WHU-CD', 'CLCD', 
        'CaBuAr-CD', 'S2Looking-CD', 'SEN1Floods11-CD'
]
    

class CDDataset(Dataset):
    """Change Detection 통합 데이터셋"""

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        normalize: bool = True
    ):
        """
        Args:
            root_dir: dataset 폴더 경로
            dataset_name: 데이터셋 이름
            split: 'train', 'val', 'test'
            transform: Albumentations 변환
            normalize: 정규화 적용 여부
        """
        super().__init__()
        
        assert dataset_name in SUPPORTED_DATASETS, \
            f"Dataset {dataset_name} not supported. Choose from {SUPPORTED_DATASETS}"
        assert split in ['train', 'val', 'test'], \
            f"Split {split} not supported. Choose from ['train', 'val', 'test']"
        
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_path = os.path.join(root_dir, dataset_name, split)
        
        # 이미지 파일 리스트 생성
        self._load_file_list()
        
        # 변환 설정
        self.transform = transform
        self.normalize = normalize
        
        if self.normalize and self.transform is None:
            # 기본 정규화만 적용
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'image2': 'image'})
    
    def _load_file_list(self):
        """파일 리스트 로드"""
        # t1 폴더의 이미지 파일 목록
        img_dir = os.path.join(self.dataset_path, 't1')
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory not found: {img_dir}")
        
        # 이미지 파일 필터링 (.jpg, .png, .tif 등)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        self.file_list = [
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        self.file_list.sort()
        
        if len(self.file_list) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        print(f"Loaded {len(self.file_list)} images from {self.dataset_name}/{self.split}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                - 'img1': Tensor [3, H, W]
                - 'img2': Tensor [3, H, W]
                - 'label': Tensor [1, H, W]
                - 'filename': str
        """
        filename = self.file_list[idx]
        
        # 경로 설정
        img1_path = os.path.join(self.dataset_path, 't1', filename)
        img2_path = os.path.join(self.dataset_path, 't2', filename)
        label_path = os.path.join(self.dataset_path, 'label', filename)
        
        # 이미지 읽기
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # 라벨 읽기 (그레이스케일)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # 라벨 이진화 (0 또는 255 -> 0 또는 1)
        label = (label > 0).astype(np.float32)
        
        # 변환 적용
        if self.transform:
            transformed = self.transform(
                image=img1,
                image2=img2,
                mask=label
            )
            img1 = transformed['image']
            img2 = transformed['image2'] 
            label = transformed['mask']
        else:
            # transform이 없으면 기본 tensor 변환
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).float()
        
        # 라벨에 채널 차원 추가
        if len(label.shape) == 2:
            label = label.unsqueeze(0)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': label,
            'filename': filename
        }


def get_transforms(
    split: str = 'train',
    img_size: int = 256,
    augment: bool = True
) -> A.Compose:
    """데이터 변환 파이프라인 생성"""
    
    if split == 'train' and augment:
        transform = A.Compose([
            # 크기 조정 (이미 256x256이면 불필요)
            A.Resize(img_size, img_size),
            
            # 데이터 증강
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussNoise(p=0.1),
            
            # 정규화
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], additional_targets={'image2': 'image'})
    else:
        # Validation/Test - 증강 없음
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], additional_targets={'image2': 'image'})
    
    return transform


def create_dataloaders(
    root_dir: str,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    augment: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터로더 생성 헬퍼 함수"""
    
    # Train
    train_transform = get_transforms('train', img_size, augment)
    train_dataset = CDDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        split='train',
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    # Validation
    val_transform = get_transforms('val', img_size, augment=False)
    val_dataset = CDDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        split='val',
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Test
    test_transform = get_transforms('test', img_size, augment=False)
    test_dataset = CDDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        split='test',
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 테스트 코드
    root_dir = "../dataset"
    dataset_name = "LEVIR-CD+"
    
    # 데이터셋 생성 테스트
    dataset = CDDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 샘플 확인
    sample = dataset[0]
    print(f"Image1 shape: {sample['img1'].shape}")
    print(f"Image2 shape: {sample['img2'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Filename: {sample['filename']}")