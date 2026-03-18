import os
import csv
import time
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
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

# Get the arguments
args = get_arguments()

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
    """指标记录器，生成标准格式CSV"""
    
    def __init__(self, save_path, model_info, mode='train'):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.mode = mode
        
        if mode == 'train':
            self.header = [
                'epoch',
                'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
                'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
                'inference_time_ms', 'fps', 'learning_rate'
            ]
        else:  # test
            self.header = [
                'model_path', 'model_type', 'test_images', 'test_masks',
                'input_height', 'input_width', 'num_classes',
                'total_params', 'model_size_mb',
                'test_loss', 'test_precision', 'test_recall', 'test_f1', 'test_miou',
                'test_acc', 'test_kappa', 'inference_time_ms', 'fps', 'total_images'
            ]
        self.rows = []
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录训练轮次"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_metrics.get('precision', 0):.6f}",
            'train_recall': f"{train_metrics.get('recall', 0):.6f}",
            'train_f1': f"{train_metrics.get('f1', 0):.6f}",
            'train_miou': f"{train_metrics.get('miou', 0):.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_precision': f"{val_metrics.get('precision', 0):.6f}",
            'val_recall': f"{val_metrics.get('recall', 0):.6f}",
            'val_f1': f"{val_metrics.get('f1', 0):.6f}",
            'val_miou': f"{val_metrics.get('miou', 0):.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        self.rows.append(row)
        
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
            'test_acc': f"{metrics['acc']:.6f}",
            'test_kappa': f"{metrics['kappa']:.6f}",
            'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
            'fps': f"{metrics['fps']:.2f}",
            'total_images': metrics['total_images'],
        }
        self.rows.append(row)
        
    def save(self):
        """保存CSV"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Metrics log saved to {self.save_path}")
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model Type: ENet\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            if self.mode == 'train':
                f.write(f"Epochs: {args.epochs}\n")
                f.write(f"Learning Rate: {args.learning_rate}\n")
                f.write(f"Batch Size: {args.batch_size}\n")


def compute_metrics_from_iou(iou_list, num_classes):
    """从IoU计算其他指标（简化版）"""
    # 这里使用IoU作为近似，实际需要根据混淆矩阵计算
    # 由于ENet的IoU类没有直接提供Precision/Recall，我们使用IoU近似
    miou = np.mean(iou_list)
    
    # 简化：假设Precision/Recall/F1与IoU相近（实际应该修改IoU类提供这些指标）
    return {
        'precision': miou,  # 近似
        'recall': miou,     # 近似
        'f1': miou,         # 近似
        'miou': miou,
    }


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

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
            print("Train images:", args.images)
            print("Train masks:", args.masks)
            print("Val images:", args.val_images)
            print("Val masks:", args.val_masks)
            
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
                print("Test images:", args.test_images)
                print("Test masks:", args.test_masks)
                test_set = dataset(
                    root_dir=args.test_images,
                    transform=image_transform,
                    label_transform=label_transform_water,
                    mask_dir=args.test_masks
                )
            else:
                print("No test set provided, using val set as test set")
                test_set = val_set
                
        elif manual_train and not manual_val:
            print("\n" + "="*50)
            print(">>> Using AUTO split mode (from manual train set) <<<")
            print("="*50)
            print("Train images:", args.images)
            print("Train masks:", args.masks)
            print(f"Auto-splitting with val_split={args.val_split}, seed={args.seed}")
            
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
            
            if val_size == 0 or train_size == 0:
                raise ValueError(f"Cannot split dataset: train={train_size}, val={val_size}. "
                               f"Check your val_split ({val_split}) and dataset size ({dataset_size})")
            
            print(f"Auto-splitting dataset: {train_size} train, {val_size} val")
            
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
            print("Dataset dir:", args.dataset_dir)
            print(f"Auto-splitting with val_split={args.val_split}, seed={args.seed}")
            
            if not os.path.exists(args.dataset_dir):
                raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}. "
                                      f"Please provide --dataset-dir or use --images and --masks")
            
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
            
            if val_size == 0 or train_size == 0:
                raise ValueError(f"Cannot split dataset: train={train_size}, val={val_size}. "
                               f"Check your val_split ({val_split}) and dataset size ({dataset_size})")
            
            print(f"Auto-splitting dataset: {train_size} train, {val_size} val")
            
            train_set, val_set = data.random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            test_set = val_set
        
    else:
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
    print("Test dataset size:", len(test_set))

    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))
    else:
        images, labels = next(iter(train_loader))
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    
    class_weights = 0
    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_classes)
    elif args.weighing.lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    model = ENet(num_classes).to(device)
    print(model)

    # 获取模型信息并创建记录器
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 创建记录器
    log_path = Path(args.save_dir) / f"enet_training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_logger = MetricsLogger(log_path, model_info, mode='train')
    metrics_logger.save_model_info()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    print()
    train_runner = Train(model, train_loader, optimizer, criterion, metric, device)
    val_runner = Test(model, val_loader, criterion, metric, device)
    
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        # 训练阶段
        epoch_start = time.time()
        epoch_loss, (iou, miou) = train_runner.run_epoch(args.print_step)
        train_time = time.time() - epoch_start
        
        # 计算训练指标
        # train_metrics = compute_metrics_from_iou(iou, num_classes)
        train_metrics = metric.compute_metrics() # 使用新方法
        
        lr_updater.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))

        # 验证阶段
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            val_start = time.time()
            val_loss, (val_iou, val_miou) = val_runner.run_epoch(args.print_step)
            val_time = time.time() - val_start
            
            # 计算验证指标和推理时间
            # val_metrics = compute_metrics_from_iou(val_iou, num_classes)
            val_metrics = metric.compute_metrics() # 使用新方法
            
            # 计算FPS（近似）
            val_dataset_size = len(val_loader.dataset)
            fps = val_dataset_size / val_time if val_time > 0 else 0
            inference_time_ms = (val_time / len(val_loader)) * 1000 if len(val_loader) > 0 else 0

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, val_loss, val_miou))

            if epoch + 1 == args.epochs or val_miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), val_iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            if val_miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = val_miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, args)
            
            # 记录到CSV
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

    # 保存训练日志
    metrics_logger.save()
    print(f"\nTraining complete! Best Val mIoU: {best_miou:.4f}")

    return model


def test(model, test_loader, class_weights, class_encoding, test_images=None, test_masks=None):
    print("\nTesting...\n")

    num_classes = len(class_encoding)
    
    # 获取模型信息
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.dataset.lower() == 'water':
        ignore_index = None
    elif args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    test_runner = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")
    
    # 测量推理时间
    test_start = time.time()
    loss, (iou, miou) = test_runner.run_epoch(args.print_step)
    test_time = time.time() - test_start
    
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    for key, class_iou_val in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou_val))

    # 计算指标
    # metrics = compute_metrics_from_iou(iou, num_classes)
    metrics = metric.compute_metrics() # 使用新方法
    metrics['loss'] = loss
    metrics['acc'] = miou  # ENet没有直接提供Acc，用mIoU近似
    metrics['kappa'] = miou  # 近似
    
    # 计算推理时间
    test_dataset_size = len(test_loader.dataset)
    fps = test_dataset_size / test_time if test_time > 0 else 0
    inference_time_ms = (test_time / len(test_loader)) * 1000 if len(test_loader) > 0 else 0
    
    metrics['inference_time_ms'] = inference_time_ms
    metrics['fps'] = fps
    metrics['total_images'] = test_dataset_size

    # 确定测试数据路径
    if test_images is None:
        test_images = getattr(args, 'test_images', args.images if args.images else args.dataset_dir)
    if test_masks is None:
        test_masks = getattr(args, 'test_masks', args.masks if args.masks else 'N/A')

    # 创建记录器并保存
    log_path = Path(args.save_dir) / f"enet_test_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_logger = MetricsLogger(log_path, model_info, mode='test')
    
    # 尝试获取模型路径
    model_path = getattr(args, 'resume', 'unknown')
    if model_path and os.path.exists(os.path.join(args.save_dir, f"{args.name}.pth")):
        model_path = os.path.join(args.save_dir, f"{args.name}.pth")
    
    metrics_logger.log_test(
        model_path if model_path else args.save_dir,
        test_images,
        test_masks,
        (args.height, args.width),
        num_classes,
        metrics,
        model_info
    )
    metrics_logger.save()

    return metrics


def predict(model, images, class_encoding):
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images)

    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


if __name__ == '__main__':

    assert os.path.isdir(args.save_dir), "The directory \"{0}\" doesn't exist.".format(args.save_dir)

    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'water':
        from data.water_dataset import WaterDataset as dataset
        
        if args.images is None and args.masks is None and args.dataset_dir is None:
            raise ValueError("For water dataset, please provide either --dataset-dir or --images and --masks")
        if args.images is None and args.masks is None:
            assert os.path.isdir(args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(args.dataset_dir)
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
            raise NotImplementedError("Test mode for non-water datasets requires dataset-specific handling")
        
        final_test_loader = None
        model = None

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        optimizer = optim.Adam(model.parameters())

        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

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
                    print("\n" + "="*50)
                    print(">>> Using SPECIFIED test set <<<")
                    print("="*50)
                    
                    test_set = dataset(
                        root_dir=args.test_images,
                        transform=image_transform,
                        label_transform=label_transform_water,
                        mask_dir=args.test_masks
                    )
                    test_images_path = args.test_images
                    test_masks_path = args.test_masks
                    
                elif args.images is not None and args.masks is not None:
                    print("\n" + "="*50)
                    print(">>> Using specified images/masks as test set <<<")
                    print("="*50)
                    
                    test_set = dataset(
                        root_dir=args.images,
                        transform=image_transform,
                        label_transform=label_transform_water,
                        mask_dir=args.masks
                    )
                    test_images_path = args.images
                    test_masks_path = args.masks
                    
                elif args.dataset_dir is not None:
                    print("\n" + "="*50)
                    print(">>> Using AUTO test set from dataset-dir <<<")
                    print("="*50)
                    
                    full_dataset = dataset(
                        args.dataset_dir,
                        transform=image_transform,
                        label_transform=label_transform_water,
                        mask_dir=None
                    )
                    
                    dataset_size = len(full_dataset)
                    test_size = int(args.val_split * dataset_size)
                    train_val_size = dataset_size - test_size
                    
                    _, test_set = data.random_split(
                        full_dataset,
                        [train_val_size, test_size],
                        generator=torch.Generator().manual_seed(args.seed)
                    )
                    test_images_path = args.dataset_dir
                    test_masks_path = args.dataset_dir
                else:
                    raise ValueError("No test data specified")
                
                final_test_loader = data.DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers
                )
                
                print(f"Test dataset size: {len(test_set)}")
            else:
                raise NotImplementedError("Test mode for non-water datasets not implemented")
        
        test_loader_to_use = final_test_loader if final_test_loader is not None else test_loader
        
        test(model, test_loader_to_use, w_class, class_encoding, 
             test_images=test_images_path if 'test_images_path' in locals() else None,
             test_masks=test_masks_path if 'test_masks_path' in locals() else None)