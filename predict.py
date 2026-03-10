import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.enet import ENet
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='ENet Water Segmentation Prediction')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the trained model checkpoint (directory or .pkl file)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--height', type=int, default=360,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=480,
                        help='Input image width')
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save overlay visualization (image + red mask)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for overlay (0-1)')
    return parser.parse_args()

def load_model(model_path, device):
    """加载训练好的模型"""
    model = ENet(num_classes=2).to(device)
    
    # 处理目录路径
    if os.path.isdir(model_path):
        pkl_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {model_path}")
        model_path = os.path.join(model_path, pkl_files[0])
        print(f"Found checkpoint: {model_path}")
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    # 提取模型权重
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}, mIoU: {checkpoint.get('miou', 0):.4f}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def predict_image(model, image_path, transform, device):
    """对单张图片进行预测"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0)
    
    # 转为 PIL，调整回原始尺寸
    pred_mask = pred.cpu().numpy().astype(np.uint8)
    pred_mask_pil = Image.fromarray(pred_mask * 255)  # 0->0, 1->255
    pred_mask_pil = pred_mask_pil.resize(original_size, Image.NEAREST)
    
    return image, pred_mask_pil, pred_mask

def create_overlay(image, mask, alpha=0.5):
    """创建红色半透明叠加图"""
    # 转为 numpy
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask)
    
    # 确保 mask 是 2D
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]
    
    # 创建红色叠加层 (RGB: 255, 0, 0)
    overlay = img_array.copy()
    overlay[mask_array > 128] = [255, 0, 0]  # 红色标记水域区域
    
    # 混合: 原图 * (1-alpha) + 红色叠加层 * alpha
    blended = img_array * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def main():
    args = get_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_overlay:
        overlay_dir = os.path.join(args.output_dir, 'overlay')
        os.makedirs(overlay_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, device)
    
    # 图像预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])
    
    # 获取所有图片
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(args.input_dir) 
                   if f.lower().endswith(valid_ext)]
    image_files.sort()
    
    print(f"Found {len(image_files)} images in {args.input_dir}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # 批量预测
    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(args.input_dir, img_name)
        
        try:
            # 预测
            original_image, pred_mask, pred_array = predict_image(
                model, img_path, transform, device
            )
            
            base_name = os.path.splitext(img_name)[0]
            
            # 保存黑白 mask
            mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
            pred_mask.save(mask_path)
            
            # 保存红色叠加图
            if args.save_overlay:
                overlay = create_overlay(original_image, pred_mask, args.alpha)
                overlay_path = os.path.join(args.output_dir, 'overlay', 
                                          f"{base_name}_overlay.png")
                overlay.save(overlay_path)
            
            # 计算水域占比
            water_ratio = np.mean(pred_array) * 100
            tqdm.write(f"{img_name}: Water area = {water_ratio:.2f}%")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Done! Results saved to: {args.output_dir}")
    print(f"   - *_mask.png: Binary mask (black=background, white=water)")
    if args.save_overlay:
        print(f"   - overlay/*_overlay.png: Original image with RED water overlay")

if __name__ == '__main__':
    main()