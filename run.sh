#训练模型
python main_water.py -m train `
    --save-dir save/folder/ `
    --name model_name `
    --dataset name `
    --dataset-dir path/root_directory/

#训练模型（继续）
python main_water.py -m train `
    --resume True `
    --save-dir save/folder/ `
    --name model_name `
    --dataset name `
    --dataset-dir path/root_directory/

#训练模型（水域分割）
python main_water.py `
    --mode train `
    --dataset water `
    --dataset-dir ./data/WaterSegmentDataset `
    --save-dir ./save `
    --name enet_water `
    --batch-size 8 `
    --epochs 100 `
    --learning-rate 0.0005 `
    --val_split 0.2 `
    --seed 42 `
    --weighing none

#训练模型（水域分割，train与val分离）
#模式1：手动指定 train 和 val✅
python main_water_val.py `
    --mode train `
    --dataset water `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_red `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_red `
    --save-dir ./save `
    --name enet_water `
    --batch-size 8 `
    --epochs 5 `
    --learning-rate 0.0005 `
    --weighing none `
    --with-unlabeled
#模式2：只给 images 和 masks，自动划分 val
python main_water_val.py `
    --mode train `
    --dataset water `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_red `
    --save-dir ./save `
    --name enet_water `
    --batch-size 8 `
    --epochs 5 `
    --learning-rate 0.0005 `
    --val_split 0.2 `
    --seed 42 `
    --weighing none
#模式3：完全自动模式
python main_water_val.py `
    --mode train `
    --dataset water `
    --dataset-dir ./data/WaterSegmentDataset `
    --save-dir ./save `
    --name enet_water `
    --batch-size 8 `
    --epochs 5 `
    --learning-rate 0.0005 `
    --val_split 0.2 `
    --seed 42 `
    --weighing none

#模型训练（pro）
python main_water_val_pro.py `
    --mode train `
    --dataset water `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_red `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_red `
    --model-dir ./checkpoints `
    --log-dir ./logs `
    --model-name enet_exp01 `
    --log-name enet_exp01_log `
    --save-interval 0 `
    --batch-size 4 `
    --epochs 5 `
    --learning-rate 5e-4 `
    --weighing none `
    --with-unlabeled

#测试模型
python main_water.py -m test `
    --save-dir save/folder/ `
    --name model_name `
    --dataset name `
    --dataset-dir path/root_directory/

#测试模型（水域分割）
python main_water.py -m test `
    --save-dir ./save `
    --name enet_water `
    --dataset water `
    --dataset-dir ./data/WaterSegmentDataset
#测试模型（水域分割，）
# 方式1：指定test路径（推荐）✅
python main_water_val.py -m test `
    --save-dir ./save `
    --name enet_water `
    --dataset water `
    --test-images D:\Files\Data\IRWSB\test\images `
    --test-masks D:\Files\Data\IRWSB\test\masks_red
# 方式2：使用--images和--masks（如果没有指定--test-images）
python main_water_val.py -m test `
    --save-dir ./save `
    --name enet_water `
    --dataset water `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_red
# 方式3：原有的自动模式（仍然支持）
python main_water_val.py -m test `
    --save-dir ./save `
    --name enet_water `
    --dataset water `
    --dataset-dir ./data/WaterSegmentDataset


#预测模型
# 基本用法：只生成mask
python predict.py `
    --model-path ./save/enet_water/enet_water.pkl `
    --input-dir ./data/WaterSegmentDataset/images `
    --output-dir ./predictions

# 带可视化叠加图（推荐，红色标记水域）
python predict.py `
    --model-path ./save/enet_water `
    --input-dir ./data/img `
    --output-dir ./data/predictions `
    --save-overlay `
    --alpha 0.5

#预测模型pro（生成红色mask蒙版和csv评价指标）
python predict_pro.py `
    --input "D:\Files\Data\IRWSB\analyse\images" `
    --model-path "F:\AAA\7_enetbest\experiment1\experiment1_last.pth" `
    --output ./results `
    --ground-truth "D:\Files\Data\IRWSB\analyse\masks_red" `
    --debug
