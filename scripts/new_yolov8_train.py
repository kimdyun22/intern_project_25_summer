import torch
import os
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob

# ======== 설정 ========
project_name = "brake_disc_cls_augmented"
project_root = "/workspace"
base_dir = os.path.join(project_root, "classify", project_name)
weights_dir = os.path.join(base_dir, "weights")
output_fig_dir = base_dir
os.makedirs(output_fig_dir, exist_ok=True)

# GPU 확인
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"✅ Device: {'GPU' if device == 0 else 'CPU'}")

# ======== 모델 초기화 ========
model = YOLO("yolov8m-cls.pt")  # 사전학습된 분류 모델 사용

# ======== 학습 ========
data_path = "/workspace/dataset_split"
model.train(
    data=data_path,
    epochs=100,
    imgsz=64,
    batch=32,
    device=device,
    name=project_name,
    project="/workspace",

    # augmentation
    fliplr=0.5,
    flipud=0.2,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.4,
    auto_augment="randaugment",

    # "콜백 역할" 파라미터 👇
    patience=10,       # early stopping (val 성능 정체 시 10 epoch 후 중단)
    cos_lr=True,       # cosine learning rate schedule
    lrf=0.01,          # final learning rate factor (최종 lr = lr0 * lrf)
)



# ======== 테스트 평가 (수정됨) ========
test_paths = glob.glob(os.path.join(data_path, "test", "*", "*.bmp"))

if len(test_paths) == 0:
    raise FileNotFoundError("❌ 테스트 이미지가 없습니다. '/workspace/dataset_split/test/*/*.bmp' 경로를 확인하세요.")

results = model.predict(source=test_paths, imgsz=224, device=device, verbose=False)

# 결과 분석
y_true, y_pred = [], []
for r in results:
    pred_class = r.names[r.probs.top1]
    filename = os.path.basename(r.path)
    label_token = filename.split('_')[-1].split('.')[0]
    if label_token == "OK":
        true_class = "OK"
    elif label_token == "NG":
        true_class = "NG"
    else:
        true_class = "unknown"
    y_pred.append(pred_class)
    y_true.append(true_class)

# 정확도 및 혼동 행렬
labels = sorted(list(set(y_true + y_pred)))
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (Acc: {acc * 100:.2f}%)")
plt.savefig(os.path.join(output_fig_dir, "confusion_matrix.png"))
plt.close()

# 정확도 그래프
# Accuracy curve
results_csv = os.path.join(base_dir, "results.csv")
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)

    # 정확도 컬럼 자동 탐색
    acc_columns = [col for col in df.columns if "acc" in col.lower()]
    if acc_columns:
        acc_col = acc_columns[0]  # 첫 번째 정확도 컬럼 사용
        fig_acc = plt.figure()
        sns.lineplot(x=df["epoch"], y=df[acc_col], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy over Epochs")
        plt.grid(True)
        plt.savefig(os.path.join(output_fig_dir, "accuracy_curve.png"))
        plt.close()
    else:
        print("⚠️ 정확도 관련 컬럼을 찾을 수 없습니다. Accuracy curve 생략.")
else:
    print("⚠️ results.csv not found. Accuracy curve skipped.")


# 완료 메시지
print("\n✅ All done!")
print(f"📁 best.pt → {os.path.join(weights_dir, 'best.pt')}")
print(f"📊 Confusion matrix → {output_fig_dir}/confusion_matrix.png")
print(f"📈 Accuracy curve → {output_fig_dir}/accuracy_curve.png")
