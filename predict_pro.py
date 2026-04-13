import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.enet import ENet
import argparse
from tqdm import tqdm
import csv
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description='ENet Water Segmentation Prediction')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory containing images')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint (directory or .pkl file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save prediction overlay results')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Directory containing ground truth masks (for evaluation)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--height', type=int, default=360,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=480,
                        help='Input image width')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for overlay (0-1)')
    parser.add_argument('--save-mask', action='store_true',
                        help='Also save binary mask (black=background, white=water)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()

def load_model(model_path, device):
    """加载训练好的模型"""
    model = ENet(num_classes=2).to(device)
    
    if os.path.isdir(model_path):
        pkl_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {model_path}")
        model_path = os.path.join(model_path, pkl_files[0])
        print(f"Found checkpoint: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
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
    """对单张图片进行预测，返回原图、预测mask、resize后的预测结果、推理时间(ms)"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)

    input_tensor = transform(image).unsqueeze(0).to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    inference_time_ms = (end - start) * 1000

    pred = torch.argmax(output, dim=1).squeeze(0)
    pred_mask = pred.cpu().numpy().astype(np.uint8)
    pred_mask_pil = Image.fromarray(pred_mask * 255)
    pred_mask_pil = pred_mask_pil.resize(original_size, Image.NEAREST)
    pred_mask_resized = np.array(pred_mask_pil) // 255

    return image, pred_mask_pil, pred_mask_resized, inference_time_ms

def create_overlay(image, mask, alpha=0.5):
    """创建红色半透明叠加图"""
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask)
    
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]
    
    overlay = img_array.copy()
    overlay[mask_array > 128] = [255, 0, 0]
    
    blended = img_array * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def load_ground_truth(gt_path, target_size, debug=False):
    """
    加载真值mask（支持Palette模式P、RGB、灰度L）
    黑背景，红前景
    """
    gt = Image.open(gt_path)
    
    # ==================== 关键修复 ====================
    # 如果是Palette模式，转换为RGB
    if gt.mode == 'P':
        gt = gt.convert('RGB')
    elif gt.mode != 'RGB':
        gt = gt.convert('RGB')
    # =================================================
    
    gt = gt.resize(target_size, Image.NEAREST)
    gt_array = np.array(gt)
    
    if debug:
        print(f"  GT mode after convert: RGB, shape: {gt_array.shape}")
        print(f"  RGB unique values: {np.unique(gt_array.reshape(-1, 3), axis=0)}")
    
    # 提取红色通道（红色水体）
    r = gt_array[:,:,0]
    g = gt_array[:,:,1]
    b = gt_array[:,:,2]
    
    # 红色区域：R高，G低，B低（兼容不同红色深浅）
    gt_binary = ((r > 100) & (g < 100) & (b < 100)).astype(np.uint8)
    
    if debug:
        print(f"  R channel: {r.min()}-{r.max()}, mean={r.mean():.1f}")
        print(f"  Detected water pixels: {gt_binary.sum()}")
    
    return gt_binary

def calculate_metrics(pred_mask, gt_mask):
    """计算Precision、Recall、F1、IoU"""
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'mIoU': iou
    }

def get_image_files(input_path):
    """获取输入路径下的所有图片文件"""
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(valid_ext):
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                 if f.lower().endswith(valid_ext)]
        files.sort()
        return files
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

def find_ground_truth(gt_dir, base_name):
    """查找真值mask文件"""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        gt_path = os.path.join(gt_dir, f"{base_name}{ext}")
        if os.path.exists(gt_path):
            return gt_path
    return None

def save_csv_report(results, output_path):
    """保存CSV报告"""
    csv_path = os.path.join(output_path, 'evaluation_results.csv')
    has_metrics = any('metrics' in r for r in results)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if has_metrics:
            writer.writerow(['Image', 'Precision', 'Recall', 'F1', 'mIoU',
                             'Pred_Water(%)', 'GT_Water(%)', 'Inference_Time(ms)', 'FPS'])

            all_metrics = defaultdict(list)
            inference_times = []
            for result in results:
                if 'metrics' in result:
                    m = result['metrics']
                    writer.writerow([
                        result['name'],
                        f"{m['Precision']:.4f}",
                        f"{m['Recall']:.4f}",
                        f"{m['F1']:.4f}",
                        f"{m['mIoU']:.4f}",
                        f"{result['water_ratio']:.2f}",
                        f"{result.get('gt_water_ratio', 0):.2f}",
                        f"{result['inference_time']:.2f}",
                        f"{result['fps']:.2f}"
                    ])
                    for key in ['Precision', 'Recall', 'F1', 'mIoU']:
                        all_metrics[key].append(m[key])
                    inference_times.append(result['inference_time'])

            if all_metrics['Precision']:
                avg_infer = np.mean(inference_times)
                avg_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
                writer.writerow([])
                writer.writerow([
                    'Average',
                    f"{np.mean(all_metrics['Precision']):.4f}",
                    f"{np.mean(all_metrics['Recall']):.4f}",
                    f"{np.mean(all_metrics['F1']):.4f}",
                    f"{np.mean(all_metrics['mIoU']):.4f}",
                    '', '',
                    f"{avg_infer:.2f}",
                    f"{avg_fps:.2f}"
                ])
        else:
            writer.writerow(['Image', 'Water_Ratio(%)', 'Inference_Time(ms)', 'FPS'])
            inference_times = []
            for result in results:
                writer.writerow([
                    result['name'],
                    f"{result['water_ratio']:.2f}",
                    f"{result['inference_time']:.2f}",
                    f"{result['fps']:.2f}"
                ])
                inference_times.append(result['inference_time'])

            if inference_times:
                avg_infer = np.mean(inference_times)
                avg_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
                writer.writerow([])
                writer.writerow(['Average', '', f"{avg_infer:.2f}", f"{avg_fps:.2f}"])

    print(f"📊 CSV report saved to: {csv_path}")

def main():
    args = get_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        if args.save_mask:
            os.makedirs(os.path.join(args.output, 'masks'), exist_ok=True)
    
    model = load_model(args.model_path, device)

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])

    # Warm-up: 用 dummy 输入做一次前向，避免第一张图把 CUDA 初始化时间算进去
    dummy_input = torch.zeros(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    image_files = get_image_files(args.input)
    print(f"Found {len(image_files)} image(s)")

    all_results = []
    valid_gt_count = 0

    for idx, img_path in enumerate(tqdm(image_files, desc="Predicting")):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        debug_mode = args.debug and (idx == 0)

        try:
            original_image, pred_mask_pil, pred_array, inf_time = predict_image(
                model, img_path, transform, device
            )
            fps = 1000.0 / inf_time if inf_time > 0 else 0.0

            water_ratio = np.mean(pred_array) * 100
            result_record = {
                'name': img_name,
                'water_ratio': water_ratio,
                'inference_time': inf_time,
                'fps': fps
            }

            if args.ground_truth:
                gt_path = find_ground_truth(args.ground_truth, base_name)

                if debug_mode:
                    print(f"\n[Debug] {img_name} -> GT: {gt_path}")

                if gt_path:
                    gt_array = load_ground_truth(gt_path, original_image.size, debug=debug_mode)
                    gt_water_ratio = np.mean(gt_array) * 100
                    result_record['gt_water_ratio'] = gt_water_ratio

                    metrics = calculate_metrics(pred_array, gt_array)
                    result_record['metrics'] = metrics
                    valid_gt_count += 1

                    tqdm.write(f"{img_name}: Pred={water_ratio:.1f}%, GT={gt_water_ratio:.1f}%, "
                             f"IoU={metrics['mIoU']:.3f}, F1={metrics['F1']:.3f}, "
                             f"Time={inf_time:.2f}ms, FPS={fps:.2f}")
                else:
                    tqdm.write(f"{img_name}: Pred={water_ratio:.1f}% (GT not found), "
                             f"Time={inf_time:.2f}ms, FPS={fps:.2f}")
            else:
                tqdm.write(f"{img_name}: Water={water_ratio:.2f}%, Time={inf_time:.2f}ms, FPS={fps:.2f}")

            all_results.append(result_record)

            if args.output:
                overlay = create_overlay(original_image, pred_mask_pil, args.alpha)
                overlay.save(os.path.join(args.output, f"{base_name}_overlay.png"))

                if args.save_mask:
                    pred_mask_pil.save(os.path.join(args.output, 'masks', f"{base_name}_mask.png"))

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()

    if args.ground_truth:
        print(f"\n📈 Matched with GT: {valid_gt_count}/{len(image_files)} images")

    if args.output and all_results:
        save_csv_report(all_results, args.output)

    if args.ground_truth and all_results and any('metrics' in r for r in all_results):
        print("\n" + "="*105)
        print("EVALUATION RESULTS")
        print(f"{'Image':<30} {'Precision':<10} {'Recall':<10} {'F1':<10} {'mIoU':<10} {'Time(ms)':<10} {'FPS':<8}")
        print("-"*105)
        all_metrics = defaultdict(list)
        inference_times = []
        for result in all_results:
            if 'metrics' in result:
                m = result['metrics']
                print(f"{result['name']:<30} {m['Precision']:<10.4f} {m['Recall']:<10.4f} "
                      f"{m['F1']:<10.4f} {m['mIoU']:<10.4f} {result['inference_time']:<10.2f} {result['fps']:<8.2f}")
                for key in ['Precision', 'Recall', 'F1', 'mIoU']:
                    all_metrics[key].append(m[key])
                inference_times.append(result['inference_time'])
        if all_metrics['Precision']:
            avg_infer = np.mean(inference_times)
            avg_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
            print("-"*105)
            print(f"{'Average':<30} {np.mean(all_metrics['Precision']):<10.4f} "
                  f"{np.mean(all_metrics['Recall']):<10.4f} {np.mean(all_metrics['F1']):<10.4f} "
                  f"{np.mean(all_metrics['mIoU']):<10.4f} {avg_infer:<10.2f} {avg_fps:<8.2f}")
        print("="*105)
    elif all_results and not args.ground_truth:
        # 没有 GT 时也打印一下时间汇总
        print("\n" + "="*65)
        print("INFERENCE RESULTS")
        print(f"{'Image':<30} {'Water_Ratio(%)':<15} {'Time(ms)':<10} {'FPS':<8}")
        print("-"*65)
        inference_times = []
        for result in all_results:
            print(f"{result['name']:<30} {result['water_ratio']:<15.2f} {result['inference_time']:<10.2f} {result['fps']:<8.2f}")
            inference_times.append(result['inference_time'])
        if inference_times:
            avg_infer = np.mean(inference_times)
            avg_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
            print("-"*65)
            print(f"{'Average':<30} {'':<15} {avg_infer:<10.2f} {avg_fps:<8.2f}")
        print("="*65)

    print(f"\n✅ Done!")

if __name__ == '__main__':
    main()