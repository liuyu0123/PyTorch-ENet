import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

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

    # ========== 修改部分：支持 water 数据集的灵活划分 ==========
    
    if args.dataset.lower() == 'water':
        # Water dataset：不使用 PILToLongTensor，在 Dataset 内部处理标签
        label_transform_water = transforms.Compose([
            transforms.Resize((args.height, args.width), Image.NEAREST),
            # 注意：不要放 PILToLongTensor！
        ])
        
        # 检查是否手动指定了训练/验证路径
        manual_train = args.images is not None and args.masks is not None
        manual_val = args.val_images is not None and args.val_masks is not None
        
        if manual_train and manual_val:
            # ✅ 模式1：手动指定训练集和验证集
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
                mask_dir=args.masks  # 手动模式：传入mask_dir
            )
            
            val_set = dataset(
                root_dir=args.val_images,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=args.val_masks
            )
            
            # 测试集：如果指定了就用，否则用验证集代替
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
            # ✅ 模式2：只指定了训练集，自动划分验证集
            print("\n" + "="*50)
            print(">>> Using AUTO split mode (from manual train set) <<<")
            print("="*50)
            print("Train images:", args.images)
            print("Train masks:", args.masks)
            print(f"Auto-splitting with val_split={args.val_split}, seed={args.seed}")
            
            # 先加载完整训练集（手动模式）
            full_dataset = dataset(
                root_dir=args.images,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=args.masks  # 手动模式
            )
            
            # 自动划分
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
            # ✅ 模式3：完全自动模式（原有逻辑）
            print("\n" + "="*50)
            print(">>> Using FULL AUTO split mode <<<")
            print("="*50)
            print("Dataset dir:", args.dataset_dir)
            print(f"Auto-splitting with val_split={args.val_split}, seed={args.seed}")
            
            # 检查 dataset_dir 是否存在
            if not os.path.exists(args.dataset_dir):
                raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}. "
                                      f"Please provide --dataset-dir or use --images and --masks")
            
            full_dataset = dataset(
                args.dataset_dir,
                transform=image_transform,
                label_transform=label_transform_water,
                mask_dir=None  # 自动模式：不传入mask_dir
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
        # 原有的 CamVid/Cityscapes 逻辑（保持不变）
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

    # 创建 DataLoader
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

    # 获取类别编码
    if args.dataset.lower() == 'water':
        class_encoding = {'unlabeled': (0, 0, 0), 'water': (255, 255, 255)}
    else:
        class_encoding = train_set.color_encoding
        if args.dataset.lower() == 'camvid':
            del class_encoding['road_marking']

    num_classes = len(class_encoding)

    # 打印调试信息
    print("\nNumber of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    print("Test dataset size:", len(test_set))

    # 显示样本批次
    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))
    else:
        images, labels = next(iter(train_loader))
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # 显示批次图像
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # 计算类别权重
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

    # Intialize ENet
    model = ENet(num_classes).to(device)
    # Check if the network architecture is correct
    print(model)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)
        lr_updater.step()

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      args)

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'water':
        from data.water_dataset import WaterDataset as dataset
        
        # water数据集的特殊检查：如果使用自动模式，需要检查dataset_dir
        if args.images is None and args.masks is None:
            assert os.path.isdir(
                args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
                    args.dataset_dir)
    else:
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

        test(model, test_loader, w_class, class_encoding)