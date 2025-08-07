import os
import shutil
import random

# Docker 컨테이너 기준 경로
src_root = "/workspace/dataset"
classes = ["OK", "NG"]

dst_root = "/workspace/dataset_split"
splits = ["train", "val", "test"]
split_ratio = [0.7, 0.15, 0.15]

# 디렉토리 생성
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(dst_root, split, cls), exist_ok=True)

# 클래스별 분할
for cls in classes:
    src_dir = os.path.join(src_root, cls)
    all_files = [f for f in os.listdir(src_dir) if f.endswith(".bmp")]
    random.shuffle(all_files)

    total = len(all_files)
    n_train = int(total * split_ratio[0])
    n_val = int(total * split_ratio[1])

    split_files = {
        "train": all_files[:n_train],
        "val": all_files[n_train:n_train + n_val],
        "test": all_files[n_train + n_val:]
    }

    for split, files in split_files.items():
        for fname in files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_root, split, cls, fname)
            shutil.copy2(src_path, dst_path)

print("✅ 데이터셋 분할 완료!")
