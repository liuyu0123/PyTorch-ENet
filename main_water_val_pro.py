# encoding = utf-8

import os
import sys
import csv
import time
import datetime
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
import utils

# =============================================================================
# 新增参数预解析（使用 allow_abbrev=False 防止与 --mode 冲突）
# =============================================================================
_pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_pre_parser.add_argument('--model-dir', type=str, default=None, 
                         help='模型保存目录（默认: checkpoints）')
_pre_parser.add_argument('--log-dir', type=str, default=None, 
                         help='日志保存目录（默认: logs）')
_pre_parser.add_argument('--save-interval', type=int, default=0, 
                         help='分步保存模型的epoch间隔，0表示不保存中间模型')
_pre_parser.add_argument('--model-name', type=str, default=None, 
                         help='模型保存文件名前缀（默认使用 --name）')
_pre_parser.add_argument('--log-name', type=str, default=None, 
                         help='日志文件名前缀（默认自动生成）')

_pre_args, remaining = _pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining  # 移除已解析的参数，保留给原args.py

# 导入原参数（args.py 中定义了 --mode 等参数）
from args import get_arguments
args = get_arguments()

# 合并新参数到 args
args.model_dir = _pre_args.model_dir if _pre_args.model_dir else 'checkpoints'
args.log_dir = _pre_args.log_dir if _pre_args.log_dir else 'logs'
args.save_interval = _pre_args.save_interval
args.model_name = _pre_args.model_name if _pre_args.model_name else getattr(args, 'name', 'enet_model')
args.log_name = _pre_args.log_name

# 创建目录
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

device = torch.device(args.device)


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


class MetricsLogger:
    """指标记录器：CSV存到log_dir，模型存到model_dir"""
    
    def __init__(self, log_dir, log_name, model_info, mode='train'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.mode = mode
        
        # 确定CSV文件名
        if log_name:
            self.csv_path = self.log_dir / f'{log_name}.csv'
        else:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
            self.csv_path = self.log_dir / f'enet_training_log_{time_str}.csv'
        
        if mode == 'train':
            # 包含前景类指标的表头
            self.header = [
                'epoch',
                'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
                'train_fg_precision', 'train_fg_recall', 'train_fg_f1', 'train_fg_miou',
                'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
                'val_fg_precision', 'val_fg_recall', 'val_fg_f1', 'val_fg_miou',
                'inference_time_ms', 'fps', 'learning_rate'
            ]
        else:  # test
            self.header = [
                'model_path', 'model_type', 'test_images', 'test_masks',
                'input_height', 'input_width', 'num_classes',
                'total_params', 'model_size_mb',
                'test_loss', 'test_precision', 'test_recall', 'test_f1', 'test_miou',
                'test_fg_precision', 'test_fg_recall', 'test_fg_f1', 'test_fg_miou',
                'test_acc', 'test_kappa', 'inference_time_ms', 'fps', 'total_images'
            ]
        
        # 立即创建文件并写入表头（防止训练中断后无文件）
        if mode == 'train':
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writeheader()
            print(f"[INFO] 训练日志文件创建: {self.csv_path}")
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录训练轮次并立即写入 CSV"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_metrics.get('precision', 0):.6f}",
            'train_recall': f"{train_metrics.get('recall', 0):.6f}",
            'train_f1': f"{train_metrics.get('f1', 0):.6f}",
            'train_miou': f"{train_metrics.get('miou', 0):.6f}",
            'train_fg_precision': f"{train_metrics.get('fg_precision', 0):.6f}",
            'train_fg_recall': f"{train_metrics.get('fg_recall', 0):.6f}",
            'train_fg_f1': f"{train_metrics.get('fg_f1', 0):.6f}",
            'train_fg_miou': f"{train_metrics.get('fg_miou', 0):.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_precision': f"{val_metrics.get('precision', 0):.6f}",
            'val_recall': f"{val_metrics.get('recall', 0):.6f}",
            'val_f1': f"{val_metrics.get('f1', 0):.6f}",
            'val_miou': f"{val_metrics.get('miou', 0):.6f}",
            'val_fg_precision': f"{val_metrics.get('fg_precision', 0):.6f}",
            'val_fg_recall': f"{val_metrics.get('fg_recall', 0):.6f}",
            'val_fg_f1': f"{val_metrics.get('fg_f1', 0):.6f}",
            'val_fg_miou': f"{val_metrics.get('fg_miou', 0):.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        
        # 立即追加写入（实时保存）
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)
        
    def log_test(self, model_path, test_images, test_masks, input_shape, 
                 num_classes, metrics, model_info):
        """记录测试结果"""
        row = {
            'model_path': model_path,
            'model_type': 'enet',
            'test_images': test_images,
            'test_masks': test_masks,
            'input_height': input_shape[0],
            'input_width': input_shape[1],
            'num_classes': num_classes,
            'total_params': model_info['total_params'],
            'model_size_mb': f"{model_info['model_size_mb']:.2f}",
            'test_loss': f"{metrics['loss']:.6f}",
            'test_precision': f"{metrics['precision']:.6f}",
            'test_recall': f"{metrics['recall']:.6f}",
            'test_f1': f"{metrics['f1']:.6f}",
            'test_miou': f"{metrics['miou']:.6f}",
            'test_fg_precision': f"{metrics.get('fg_precision', 0):.6f}",
            'test_fg_recall': f"{metrics.get('fg_recall', 0):.6f}",
            'test_fg_f1': f"{metrics.get('fg_f1', 0):.6f}",
            'test_fg_miou': f"{metrics.get('fg_miou', 0):.6f}",
            'test_acc': f"{metrics['acc']:.6f}",
            'test_kappa': f"{metrics['kappa']:.6f}",
            'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
            'fps': f"{metrics['fps']:.2f}",
            'total_images': metrics['total_images'],
        }
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            if self.mode == 'test':
                writer.writeheader()
            writer.writerow(row)
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.log_dir / f"{self.csv_path.stem}_model_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Type: ENet\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            if self.mode == 'train':
                f.write(f"Epochs: {args.epochs}\n")
                f.write(f"Learning Rate: {args.learning_rate}\n")
                f.write(f"Batch Size: {args.batch_size}\n")
                f.write(f"Model Save Dir: {args.model_dir}\n")
                f.write(f"Log Save Dir: {args.log_dir}\n")


def compute_metrics_from_confusion_matrix(cm, num_classes):
    """
    从混淆矩阵计算各项指标（修复版）
    cm: 混淆矩阵 numpy array [num_classes, num_classes]
    """
    if cm is None or cm.sum() == 0:
        return {
            'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0,
            'fg_precision': 0, 'fg_recall': 0, 'fg_f1': 0, 'fg_miou': 0
        }
    
    metrics = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp + 1e-10) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-10) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn + 1e-10) if (tp + fp + fn) > 0 else 0
        
        metrics.append({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'miou': iou
        })
    
    # 宏平均
    macro_precision = np.mean([m['precision'] for m in metrics])
    macro_recall = np.mean([m['recall'] for m in metrics])
    macro_f1 = np.mean([m['f1'] for m in metrics])
    macro_miou = np.mean([m['miou'] for m in metrics])
    
    # 前景类（最后一类，假设为水域）
    fg = metrics[-1]
    
    # 诊断：检测全背景预测
    if fg['recall'] == 0:
        print(f"\n[WARNING] 前景类(类别{num_classes-1}) Recall=0，模型可能预测全为背景！")
        print("          这是训练初期的正常现象，建议增加训练 epoch。\n")
    elif fg['recall'] > 0 and fg['recall'] < 0.1:
        print(f"\n[INFO] 前景类 Recall={fg['recall']:.4f}，模型开始学习但效果仍较差。\n")
    
    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'miou': macro_miou,
        'fg_precision': fg['precision'],
        'fg_recall': fg['recall'],
        'fg_f1': fg['f1'],
        'fg_miou': fg['miou'],
    }


def extract_metrics_with_foreground(metric_obj, num_classes):
    """
    从 IoU 指标对象提取指标（修复版）
    尝试多种方式获取混淆矩阵
    """
    cm = None
    
    # === 新增：正确处理 IoU 对象的混淆矩阵存储路径 ===
    if hasattr(metric_obj, 'conf_metric'):
        # 方式0：直接访问 ConfusionMatrix 内部的 conf 数组（最快）
        if hasattr(metric_obj.conf_metric, 'conf'):
            cm = metric_obj.conf_metric.conf
        # 方式0b：或者调用 value() 方法获取副本
        elif hasattr(metric_obj.conf_metric, 'value'):
            cm = metric_obj.conf_metric.value()
    
    # === 原有的兼容逻辑作为 fallback ===
    elif hasattr(metric_obj, 'confusion_matrix'):
        cm = metric_obj.confusion_matrix
    elif hasattr(metric_obj, '_confusion_matrix'):
        cm = metric_obj._confusion_matrix
    elif hasattr(metric_obj, 'get_cm'):
        cm = metric_obj.get_cm()
    elif hasattr(metric_obj, 'get_confusion_matrix'):
        cm = metric_obj.get_confusion_matrix()
    
    # 确保 cm 是 numpy array
    if cm is not None:
        if torch.is_tensor(cm):
            cm = cm.cpu().numpy()
        cm = np.array(cm)
        
        if cm.shape == (num_classes, num_classes):
            return compute_metrics_from_confusion_matrix(cm, num_classes)
    
    # 如果无法获取混淆矩阵，返回零值并警告
    print(f"[WARNING] 无法从 metric 对象获取混淆矩阵，metric类型: {type(metric_obj)}")
    print(f"[WARNING] 已尝试属性: conf_metric.conf, confusion_matrix 等")
    return {
        'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0,
        'fg_precision': 0, 'fg_recall': 0, 'fg_f1': 0, 'fg_miou': 0
    }


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", getattr(args, 'dataset_dir', 'N/A'))
    print("Model save directory:", args.model_dir)
    print("Log save directory:", args.log_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    if args.dataset.lower() == 'water':
        label_transform_water = transforms.Compose([
            transforms.Resize((args.height, args.width), Image.NEAREST),
        ])
        
        manual_train = args.images is not None and args.masks is not None
        manual_val = args.val_images is not None and args.val_masks is not None
        
        if manual_train and manual_val:
            print("\n" + "="*50)
            print(">>> Using MANUAL split mode (train + val) <<<")
            print("="*50)
            
            train_set = dataset(
                root_dir=args.images,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=args.masks
            )
            
            val_set = dataset(
                root_dir=args.val_images,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=args.val_masks
            )
            
            if args.test_images is not None and args.test_masks is not None:
                test_set = dataset(
                    root_dir=args.test_images,
                    transform=image_transform,
                    label_transform=label_transform_water,
                    mask_dir=args.test_masks
                )
            else:
                test_set = val_set
                
        elif manual_train and not manual_val:
            print("\n" + "="*50)
            print(">>> Using AUTO split mode <<<")
            print("="*50)
            
            full_dataset = dataset(
                root_dir=args.images,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=args.masks
            )
            
            val_split = args.val_split
            seed = args.seed
            dataset_size = len(full_dataset)
            val_size = int(val_split * dataset_size)
            train_size = dataset_size - val_size
            
            train_set, val_set = data.random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            test_set = val_set
            
        else:
            print("\n" + "="*50)
            print(">>> Using FULL AUTO split mode <<<")
            print("="*50)
            
            full_dataset = dataset(
                args.dataset_dir,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=None
            )
            
            val_split = args.val_split
            seed = args.seed
            dataset_size = len(full_dataset)
            val_size = int(val_split * dataset_size)
            train_size = dataset_size - val_size
            
            train_set, val_set = data.random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            test_set = val_set
        
    else:
        # 其他数据集（CamVid, Cityscapes）
        train_set = dataset(
            args.dataset_dir,
            transform=image_transform,
            label_transform=label_transform)
        val_set = dataset(
            args.dataset_dir,
            mode='val',
            transform=image_transform,
            label_transform=label_transform)
        test_set = dataset(
            args.dataset_dir,
            mode='test',
            transform=image_transform,
            label_transform=label_transform)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    if args.dataset.lower() == 'water':
        class_encoding = {'unlabeled': (0, 0, 0), 'water': (255, 255, 255)}
    else:
        class_encoding = train_set.color_encoding
        if args.dataset.lower() == 'camvid':
            del class_encoding['road_marking']

    num_classes = len(class_encoding)

    print("\nNumber of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # 计算类别权重
    print("\nWeighing technique:", args.weighing)
    class_weights = 0
    
    if args.weighing.lower() == 'enet':
        from data.utils import enet_weighing
        class_weights = enet_weighing(train_loader, num_classes)
    elif args.weighing.lower() == 'mfb':
        from data.utils import median_freq_balancing
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    return (train_loader, val_loader, test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    model = ENet(num_classes).to(device)
    print(model)

    # 获取模型信息并创建记录器
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 创建记录器（CSV存到log_dir）
    metrics_logger = MetricsLogger(args.log_dir, args.log_name, model_info, mode='train')
    metrics_logger.save_model_info()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    
    # 创建 IoU 指标对象
    metric = IoU(num_classes, ignore_index=ignore_index)

    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    print()
    train_runner = Train(model, train_loader, optimizer, criterion, metric, device)
    val_runner = Test(model, val_loader, criterion, metric, device)
    
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        # 训练阶段 - 每轮重置metric
        metric.reset()  # 关键：重置混淆矩阵
        
        epoch_start = time.time()
        epoch_loss, (iou, miou) = train_runner.run_epoch(args.print_step)
        train_time = time.time() - epoch_start
        
        # 计算训练指标（包含前景类）
        train_metrics = extract_metrics_with_foreground(metric, num_classes)
        
        lr_updater.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(">>>> [Epoch: {0:d}] Loss: {1:.4f} | mIoU: {2:.4f} | FG Recall: {3:.4f}".format(
            epoch, epoch_loss, train_metrics.get('miou', 0), train_metrics.get('fg_recall', 0)))

        # 验证阶段 - 重置metric
        metric.reset()  # 关键：重置混淆矩阵
        
        print(">>>> [Epoch: {0:d}] Validation".format(epoch))

        val_start = time.time()
        val_loss, (val_iou, val_miou) = val_runner.run_epoch(args.print_step)
        val_time = time.time() - val_start
        
        # 计算验证指标（包含前景类）
        val_metrics = extract_metrics_with_foreground(metric, num_classes)
        # print(f"[DEBUG] Confusion matrix sum: {metric.conf_metric.conf.sum()}")
        # print(f"[DEBUG] Confusion matrix shape: {metric.conf_metric.conf.shape}")
        
        # 计算FPS
        val_dataset_size = len(val_loader.dataset)
        fps = val_dataset_size / val_time if val_time > 0 else 0
        inference_time_ms = (val_time / len(val_loader)) * 1000 if len(val_loader) > 0 else 0

        print(">>>> [Epoch: {0:d}] Val Loss: {1:.4f} | Val mIoU: {2:.4f} | Val FG Recall: {3:.4f}".format(
            epoch, val_loss, val_metrics.get('miou', 0), val_metrics.get('fg_recall', 0)))

        # 实时记录 CSV（立即写入 logs 目录）
        metrics_logger.log_epoch(
            epoch + 1, 
            epoch_loss, 
            train_metrics,
            val_loss, 
            val_metrics,
            inference_time_ms,
            fps,
            current_lr
        )

        # 保存最佳模型（存到 model_dir/checkpoints）
        if val_metrics.get('miou', 0) > best_miou:
            print("\n>>>> Best model thus far. Saving...\n")
            best_miou = val_metrics['miou']
            best_path = os.path.join(args.model_dir, f"{args.model_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'miou': best_miou,
                'optimizer': optimizer.state_dict(),
            }, best_path)
            print(f"[BEST] 模型已保存: {best_path}, mIoU: {best_miou:.4f}")
        
        # 分步保存中间模型
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            periodic_path = os.path.join(args.model_dir, f"{args.model_name}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'miou': val_metrics.get('miou', 0),
                'optimizer': optimizer.state_dict(),
            }, periodic_path)
            print(f"[CHECKPOINT] 中间模型已保存: {periodic_path}")

    # 训练结束，保存最终模型
    last_path = os.path.join(args.model_dir, f"{args.model_name}_last.pth")
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'miou': val_metrics.get('miou', 0) if 'val_metrics' in locals() else 0,
        'optimizer': optimizer.state_dict(),
    }, last_path)
    print(f"[LAST] 最终模型已保存: {last_path}")

    print(f"\n{'='*50}")
    print(f"训练完成!")
    print(f"最佳验证 mIoU: {best_miou:.4f}")
    print(f"最终验证 FG-Recall: {val_metrics.get('fg_recall', 0):.4f}")
    print(f"最佳模型: {args.model_dir}/{args.model_name}_best.pth")
    print(f"最终模型: {args.model_dir}/{args.model_name}_last.pth")
    print(f"训练日志: {metrics_logger.csv_path}")
    print(f"{'='*50}")

    return model


def test(model, test_loader, class_weights, class_encoding, test_images=None, test_masks=None):
    print("\nTesting...\n")

    num_classes = len(class_encoding)
    
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    metric = IoU(num_classes)

    test_runner = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")
    
    # 重置metric
    metric.reset()
    
    # 测量推理时间
    test_start = time.time()
    loss, (iou, miou) = test_runner.run_epoch(args.print_step)
    test_time = time.time() - test_start
    
    # 计算指标（包含前景类）
    test_metrics = extract_metrics_with_foreground(metric, num_classes)
    test_metrics['loss'] = loss
    test_metrics['acc'] = miou
    test_metrics['kappa'] = miou
    
    test_dataset_size = len(test_loader.dataset)
    fps = test_dataset_size / test_time if test_time > 0 else 0
    inference_time_ms = (test_time / len(test_loader)) * 1000 if len(test_loader) > 0 else 0
    
    test_metrics['inference_time_ms'] = inference_time_ms
    test_metrics['fps'] = fps
    test_metrics['total_images'] = test_dataset_size

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # 确定测试数据路径
    if test_images is None:
        test_images = getattr(args, 'test_images', args.images if hasattr(args, 'images') else 'N/A')
    if test_masks is None:
        test_masks = getattr(args, 'test_masks', args.masks if hasattr(args, 'masks') else 'N/A')

    # 创建记录器（存到 log_dir/logs）
    test_log_name = f"{args.model_name}_test_{time.strftime('%Y%m%d_%H%M%S')}"
    metrics_logger = MetricsLogger(args.log_dir, test_log_name, model_info, mode='test')
    
    best_model_path = os.path.join(args.model_dir, f"{args.model_name}_best.pth")
    model_path = best_model_path if os.path.exists(best_model_path) else args.model_dir
    
    metrics_logger.log_test(
        model_path,
        test_images,
        test_masks,
        (args.height, args.width),
        num_classes,
        test_metrics,
        model_info
    )

    return test_metrics


if __name__ == '__main__':
    # 导入数据集
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'water':
        from data.water_dataset import WaterDataset as dataset
    else:
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))

    if args.mode.lower() in {'train', 'full'}:
        loaders, w_class, class_encoding = load_dataset(dataset)
        train_loader, val_loader, test_loader = loaders
        
        model = train(train_loader, val_loader, w_class, class_encoding)
        
        if args.mode.lower() == 'full':
            final_test_loader = test_loader
        else:
            final_test_loader = None
    else:
        if args.dataset.lower() == 'water':
            class_encoding = {'unlabeled': (0, 0, 0), 'water': (255, 255, 255)}
            w_class = torch.ones(len(class_encoding)).to(device)
        else:
            raise NotImplementedError("Test mode requires dataset-specific handling")
        
        final_test_loader = None
        model = None

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        # 加载最佳模型进行测试
        best_model_path = os.path.join(args.model_dir, f"{args.model_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"=> Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded best model for testing")

        if final_test_loader is None:
            if args.dataset.lower() == 'water':
                image_transform = transforms.Compose([
                    transforms.Resize((args.height, args.width)),
                    transforms.ToTensor()
                ])
                label_transform_water = transforms.Compose([
                    transforms.Resize((args.height, args.width), Image.NEAREST),
                ])
                
                if args.test_images is not None and args.test_masks is not None:
                    test_set = dataset(
                        root_dir=args.test_images,
                        transform=image_transform,
                        label_transform=label_transform_water,
                        mask_dir=args.test_masks
                    )
                else:
                    raise ValueError("Please specify --test-images and --test-masks for test mode")
                
                final_test_loader = data.DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers
                )
        
        test_loader_to_use = final_test_loader if final_test_loader is not None else test_loader
        
        test(model, test_loader_to_use, w_class, class_encoding)