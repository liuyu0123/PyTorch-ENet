import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class WaterDataset(Dataset):
    """水域分割数据集"""
    
    color_encoding = {'unlabeled': (0, 0, 0), 'water': (255, 255, 255)}
    
    def __init__(self, root_dir, transform=None, label_transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.label_transform = label_transform
        
        # 获取所有图像文件
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = sorted([f for f in os.listdir(self.image_dir) 
                             if f.lower().endswith(valid_ext)])
        
        print(f"Found {len(self.images)} images in {self.image_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 标签文件名（支持多种命名方式）
        base_name = os.path.splitext(img_name)[0]
        possible_labels = [
            base_name + '_mask.png',
            base_name + '.png',
            base_name + '_label.png'
        ]
        
        label_path = None
        for label_name in possible_labels:
            temp_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(temp_path):
                label_path = temp_path
                break
        
        if label_path is None:
            raise FileNotFoundError(f"Label not found for {img_name}")
        
        # 加载图像和标签
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 灰度图
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 标签处理：先应用变换（如果有），然后统一二值化并转为tensor
        if self.label_transform:
            label = self.label_transform(label)  # 这可能是PIL Image或Tensor
        
        # 统一处理为numpy然后二值化
        if isinstance(label, Image.Image):
            label = np.array(label, dtype=np.int64)
        elif isinstance(label, torch.Tensor):
            label = label.numpy().astype(np.int64)
        
        # 二值化：0保持0，>0变为1
        label = (label > 0).astype(np.int64)
        label = torch.from_numpy(label)
            
        return image, label