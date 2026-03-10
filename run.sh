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

#测试模型
python main_water.py -m test `
    --save-dir save/folder/ `
    --name model_name `
    --dataset name `
    --dataset-dir path/root_directory/