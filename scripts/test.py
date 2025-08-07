import torch
import os
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ======== 1. 설정 ========
pt_path = "/workspace/brake_disc_cls_model2/weights/best.pt"  # 학습 완료된 .pt 파일
test_path = "/workspace/dataset_split/test_predict"  # 테스트 이미지 경로 (평탄화된 디렉토리)
output_dir = os.path.dirname(os.path.dirname(pt_path))  # classify/brake_disc_cls_model
output_fig_dir = output_dir
os.makedirs(output_fig_dir, exist_ok=True)

# ======== 2. 모델 로드 ========
model = YOLO(pt_path)

# ======== 3. 예측 수행 ========
results = model.predict(source=test_path, imgsz=224, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)

# ======== 4. 정답 및 예측 추출 ========
y_true, y_pred = [], []
for r in results:
    pred_class = r.names[r.probs.top1]
    filename = os.path.basename(r.path)
    label_token = filename.split('_')[-1].split('.')[0]  # OK or NG

    if label_token == "OK":
        true_class = "good"
    elif label_token == "NG":
        true_class = "bad"
    else:
        true_class = "unknown"

    y_pred.append(pred_class)
    y_true.append(true_class)

# ======== 5. 정확도 및 혼동 행렬 ========
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

# ======== 6. 완료 메시지 ========
print("\n✅ Evaluation Complete!")
print(f"📁 Tested weights: {pt_path}")
print(f"📊 Confusion matrix saved → {os.path.join(output_fig_dir, 'confusion_matrix_test.png')}")
