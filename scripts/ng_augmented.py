import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# 설정"/workspace/dataset_split/train/bad"
src_dir = "/workspace/dataset/NG"
dst_dir = "/workspace/dataset/NG_augmented"
os.makedirs(dst_dir, exist_ok=True)

# 증강 함수들
def augment_image(img):
    aug_images = []

    # 1. 수평 뒤집기
    aug_images.append(cv2.flip(img, 1))

    # 2. 밝기 변화
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for val in [30, -30]:
        hsv_mod = hsv.copy()
        hsv_mod[:, :, 2] = np.clip(hsv_mod[:, :, 2] + val, 0, 255)
        aug_images.append(cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR))

    # 3. 회전
    for angle in [-10, 10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        aug_images.append(rotated)

    # 4. 노이즈
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    aug_images.append(noisy)

    # 5. 확대
    scale = 1.1
    h, w = img.shape[:2]
    resized = cv2.resize(img, None, fx=scale, fy=scale)
    crop = resized[int((scale-1)*h/2):int((scale+1)*h/2), int((scale-1)*w/2):int((scale+1)*w/2)]
    crop = cv2.resize(crop, (w, h))
    aug_images.append(crop)

    return aug_images

# 증강 실행
all_files = [f for f in os.listdir(src_dir) if f.endswith(".bmp")]
counter = 0
for filename in tqdm(all_files):
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    augmented = augment_image(img)
    for i, aug in enumerate(augmented):
        new_name = f"{os.path.splitext(filename)[0]}_aug{i}.bmp"
        cv2.imwrite(os.path.join(dst_dir, new_name), aug)
        counter += 1

print(f"✅ 총 {counter}개의 증강 이미지가 저장되었습니다.")
