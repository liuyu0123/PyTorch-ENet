import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class WaterDataset(Dataset):
    """水域分割数据集
    支持两种模式：
    1. 自动模式：传入 root_dir，自动查找 images 和 labels 子文件夹
    2. 手动模式：传入 root_dir 作为 images 路径，mask_dir 作为 labels 路径
    """
    
    color_encoding = {'unlabeled': (0, 0, 0), 'water': (255, 255, 255)}
    
    def __init__(self, root_dir, transform=None, label_transform=None, mask_dir=None):
        """
        Args:
            root_dir: 在自动模式下是数据集根目录，在手动模式下是 images 目录
            transform: 图像变换
            label_transform: 标签变换
            mask_dir: 手动模式下的 masks/labels 目录路径，为 None 时启用自动模式
        """
        self.transform = transform
        self.label_transform = label_transform
        
        if mask_dir is not None:
            # ========== 手动模式：直接指定 images 和 labels 路径 ==========
            self.image_dir = root_dir
            self.label_dir = mask_dir
            
            # 确保目录存在
            if not os.path.exists(self.image_dir):
                raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
            if not os.path.exists(self.label_dir):
                raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")
            
            print(f"[Manual Mode] Using images: {self.image_dir}")
            print(f"[Manual Mode] Using labels: {self.label_dir}")
            
        else:
            # ========== 自动模式：从 root_dir 查找 images 和 labels 子文件夹 ==========
            self.root_dir = root_dir
            
            # 尝试查找 images 文件夹
            possible_image_dirs = ['images', 'image', 'img', 'data']
            self.image_dir = None
            for d in possible_image_dirs:
                temp_path = os.path.join(root_dir, d)
                if os.path.exists(temp_path):
                    self.image_dir = temp_path
                    break
            
            if self.image_dir is None:
                raise FileNotFoundError(f"Images directory not found in {root_dir}. "
                                      f"Tried: {possible_image_dirs}")
            
            # 尝试查找 labels 文件夹
            possible_label_dirs = ['labels', 'label', 'masks', 'mask', 'gt']
            self.label_dir = None
            for d in possible_label_dirs:
                temp_path = os.path.join(root_dir, d)
                if os.path.exists(temp_path):
                    self.label_dir = temp_path
                    break
            
            if self.label_dir is None:
                raise FileNotFoundError(f"Labels directory not found in {root_dir}. "
                                      f"Tried: {possible_label_dirs}")
            
            print(f"[Auto Mode] Found images: {self.image_dir}")
            print(f"[Auto Mode] Found labels: {self.label_dir}")
        
        # 获取所有图像文件
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        self.images = sorted([f for f in os.listdir(self.image_dir) 
                             if f.lower().endswith(valid_ext)])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"Found {len(self.images)} images")
        
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
            base_name + '_label.png',
            base_name + '.jpg',
            base_name + '_mask.jpg',
            base_name + '_label.jpg',
            img_name,  # 同名文件
        ]
        
        label_path = None
        for label_name in possible_labels:
            temp_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(temp_path):
                label_path = temp_path
                break
        
        if label_path is None:
            raise FileNotFoundError(f"Label not found for {img_name} in {self.label_dir}. "
                                  f"Tried: {possible_labels}")
        
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