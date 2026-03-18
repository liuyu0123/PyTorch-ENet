from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save-dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        help=("The model found in \"--save-dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes', 'water'],
        default='camvid',
        help="Dataset to use. Default: camvid (camvid/cityscapes/water)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/CamVid",
        help="Path to the root directory of the selected dataset. "
        "Default: data/CamVid")
    
    # ========== 新增：手动指定训练集路径 ==========
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to training images directory (manual mode for water dataset)")
    parser.add_argument(
        "--masks",
        type=str,
        default=None,
        help="Path to training masks/labels directory (manual mode for water dataset)")
    
    # ========== 新增：手动指定验证集路径 ==========
    parser.add_argument(
        "--val-images",
        type=str,
        default=None,
        help="Path to validation images directory (manual mode)")
    parser.add_argument(
        "--val-masks",
        type=str,
        default=None,
        help="Path to validation masks directory (manual mode)")
    
    # ========== 新增：可选的测试集路径 ==========
    parser.add_argument(
        "--test-images",
        type=str,
        default=None,
        help="Path to test images directory (optional, manual mode)")
    parser.add_argument(
        "--test-masks",
        type=str,
        default=None,
        help="Path to test masks directory (optional, manual mode)")

    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="The image width. Default: 480")
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='ENet',
        help="The class weighing technique to apply to the dataset. "
        "Default: enet")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda")

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='ENet',
        help="Name given to the model when saving. Default: ENet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save',
        help="The directory where models are saved. Default: save")

    # Auto-split settings (仅自动模式使用)
    parser.add_argument(
        '--val_split', 
        type=float, 
        default=0.2,
        help='验证集比例（默认0.2表示20%）')
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子，保证划分结果可复现')
    
    return parser.parse_args()