import os
import shutil
from tqdm import tqdm

# 设置文件路径
file_list_path = 'file_train_list.txt'
output_root = 'kitti'  # 目标根目录
source_root = 'data/kitti'      # 原始数据的根目录，根据需要修改，例如 '/data/kitti'

# 读取所有要复制的文件路径
with open(file_list_path, 'r') as f:
    lines = f.read().splitlines()

for relative_path in tqdm(lines):
    source_path = os.path.join(source_root, relative_path)
    target_path = os.path.join(output_root, relative_path)
    # 创建目标目录（如果不存在）
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # 执行复制
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
    else:
        print(f"文件不存在，跳过: {source_path}")