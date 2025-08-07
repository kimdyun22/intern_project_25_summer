import os
import cv2
import albumentations as A
from tqdm import tqdm

# 원본 및 저장 경로
input_dir = "/workspace/dataset_split/train/bad"
output_dir = "/workspace/dataset_split/train/bad_augmented"
os.makedirs(output_dir, exist_ok=True)

# 증강 파이프라인 정의
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.GaussianBlur(p=0.2),
    A.RandomGamma(p=0.3),
    A.Resize(224, 224),  # YOLOv8에 맞게 리사이즈
])

# 증강 반복 수 (1장이면 4장 만들기)
augment_per_image = 4

# bmp 이미지만 선택
bmp_files = [f for f in os.listdir(input_dir) if f.endswith(".bmp")]

for file_name in tqdm(bmp_files):
    img_path = os.path.join(input_dir, file_name)
    image = cv2.imread(img_path)

    for i in range(augment_per_image):
        augmented = augmentations(image=image)['image']
        base_name = os.path.splitext(file_name)[0]
        out_name = f"{base_name}_aug{i+1}.bmp"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, augmented)
