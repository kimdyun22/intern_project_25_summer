import os
import shutil
import random
from tqdm import tqdm

# 원본 및 증강 데이터 경로
base_dir = "/workspace/dataset"
ok_dir = os.path.join(base_dir, "OK")
ng_dir = os.path.join(base_dir, "NG")
ng_aug_dir = os.path.join(base_dir, "NG_augmented")

# 분할 비율
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 대상 클래스
classes = ['OK', 'NG']

# 출력 폴더
output_base = "/workspace/dataset_split"

# 모든 디렉토리 생성
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

# 클래스별 분할 함수
def split_and_copy(files, cls_name):
    random.shuffle(files)
    total = len(files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy(f, os.path.join(output_base, 'train', cls_name))
    for f in val_files:
        shutil.copy(f, os.path.join(output_base, 'val', cls_name))
    for f in test_files:
        shutil.copy(f, os.path.join(output_base, 'test', cls_name))

    return len(train_files), len(val_files), len(test_files)

# 1. OK 클래스 처리
ok_files = [os.path.join(ok_dir, f) for f in os.listdir(ok_dir) if f.endswith(".bmp")]
ok_train, ok_val, ok_test = split_and_copy(ok_files, "OK")

# 2. NG 클래스 처리 (원본 + 증강 포함)
ng_files = [os.path.join(ng_dir, f) for f in os.listdir(ng_dir) if f.endswith(".bmp")]
ng_aug_files = [os.path.join(ng_aug_dir, f) for f in os.listdir(ng_aug_dir) if f.endswith(".bmp")]
ng_all_files = ng_files + ng_aug_files
ng_train, ng_val, ng_test = split_and_copy(ng_all_files, "NG")

# 결과 출력
print("\n✅ 분할 완료!")
print(f"OK → train: {ok_train}, val: {ok_val}, test: {ok_test}")
print(f"NG+증강 → train: {ng_train}, val: {ng_val}, test: {ng_test}")
