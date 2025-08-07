import os
import shutil

# 컨테이너 내부 경로 기준
src_base = "/workspace/intern_datasets/차량용 브레이크 디스크 외관 검사 이미지/차량용 브레이크 디스크 외관 검사 이미지"

# 원본 양품/불량 디렉토리
src_good = os.path.join(src_base, "1. 양품 이미지 (1051ea)")
src_bad = os.path.join(src_base, "2. 불량 이미지 (218ea)")

# YOLO 분류용 경로로 변환 (good / bad)
dst_base = "/workspace/dataset"
dst_good = os.path.join(dst_base, "good")
dst_bad = os.path.join(dst_base, "bad")

# 디렉토리 생성
os.makedirs(dst_good, exist_ok=True)
os.makedirs(dst_bad, exist_ok=True)

# 이미지 복사 함수
def copy_images(src, dst):
    for file in os.listdir(src):
        if file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(src, file), os.path.join(dst, file))

copy_images(src_good, dst_good)
copy_images(src_bad, dst_bad)

print("✅ 이미지가 /workspace/dataset/good 및 bad 로 복사 완료되었습니다.")
