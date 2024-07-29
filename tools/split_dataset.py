import os
import random
import shutil
from collections import defaultdict
import json
from tqdm import tqdm

dataset_path = "where2act_dataset/where2act_pose_data_link_entity"
target_path = "where2act_dataset/faucet_0727"

# 데이터 덮어씌워지는 것 방지
os.makedirs(target_path, exist_ok=False)

dataset_dirs_dict = defaultdict(lambda:[])
for dirpath, dirname, filenames in os.walk(dataset_path):
    for filename in filenames:
        if filename == 'link_cfg.json':
            with open(os.path.join(dirpath, filename), 'r') as f:
                json_dict = json.load(f)
            obj_name = json_dict['base']['name']
            dataset_dirs_dict[obj_name].append(dirpath)

for obj_name, dataset_dirs in tqdm(dataset_dirs_dict.items()):
    
    random.shuffle(list(range(len(dataset_dirs))))
    nums = len(dataset_dirs)

    test_num = int(0.2 * nums)
    test_dirs = dataset_dirs[-test_num:]
    trainval_dirs = dataset_dirs[:-test_num]

    for test_dir in test_dirs:
        assert test_dir.split('/')[-1].isdigit(), test_dir
        shutil.copytree(test_dir, os.path.join(target_path, 'test', obj_name, test_dir.split('/')[-1]))

    trainval_nums = len(trainval_dirs)
    val_num = int(0.2 * trainval_nums)
    val_dirs = trainval_dirs[-val_num:]
    train_dirs = trainval_dirs[:-val_num]


    for train_dir in train_dirs:
        assert train_dir.split('/')[-1].isdigit(), train_dir
        shutil.copytree(train_dir, os.path.join(target_path, 'train', obj_name, train_dir.split('/')[-1]))


    for val_dir in val_dirs:
        assert val_dir.split('/')[-1].isdigit(), val_dir
        shutil.copytree(val_dir, os.path.join(target_path, 'val', obj_name,  val_dir.split('/')[-1]))

