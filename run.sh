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