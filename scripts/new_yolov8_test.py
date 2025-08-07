import torch
import os
import glob
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ======== 설정 ========
project_name = "brake_disc_cls_augmented"
project_root = "/workspace/classify"
weights_path = os.path.join(project_root, project_name, "weights", "best.pt")
data_dir = "/workspace/dataset_split/test"
output_fig_dir = os.path.join(project_root, project_name)
os.makedirs(output_fig_dir, exist_ok=True)

# GPU 확인
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"✅ Device: {'GPU' if device == 0 else 'CPU'}")

# ======== 모델 로드 ========
model = YOLO(weights_path)

# ======== 테스트 데이터 ========
test_images = glob.glob(os.path.join(data_dir, "*", "*.bmp"))
if len(test_images) == 0:
    raise FileNotFoundError(f"❌ 테스트 이미지가 없습니다: {data_dir}")
print(f"🔍 {len(test_images)}개 테스트 이미지 로드 완료")

# ======== 예측 ========
results = model.predict(source=test_images, imgsz=224, device=device, verbose=False)

# ======== 결과 처리 ========
y_true, y_pred = [], []
for r in results:
    pred_class = r.names[r.probs.top1]
    filename = os.path.basename(r.path)

    # 파일명에서 레이블 추출
    label_token = filename.split('_')[-1].split('.')[0].upper()
    if label_token in ["OK", "NG"]:
        true_class = label_token
    else:
        true_class = "unknown"

    y_pred.append(pred_class)
    y_true.append(true_class)

# ======== 정확도 및 혼동 행렬 ========
labels = sorted(list(set(y_true + y_pred)))
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# 혼동 행렬 저장
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (Acc: {acc * 100:.2f}%)")
plt.savefig(os.path.join(output_fig_dir, "confusion_matrix_test.png"))
plt.close()

# ======== 완료 메시지 ========
print("📊 혼동 행렬 저장 완료 →", os.path.join(output_fig_dir, "confusion_matrix_test.png"))
print("✅ 테스트 평가 종료")
