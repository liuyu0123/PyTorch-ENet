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