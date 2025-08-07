import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

# 설정
DATA_DIR = "/workspace/dataset_split/test"
CHECKPOINT_PATH = "vit_brake_classifier.pth"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 테스트 데이터셋
test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 모델 정의 및 로드
model = create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 평가 루프
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Test"):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 결과 출력
print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# 저장 디렉토리 생성
SAVE_DIR = "/workspace/results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 클래스 이름
class_names = test_dataset.classes  # 예: ['OK', 'NG']

# -------------------------------
# 1. Confusion Matrix 저장
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax_cm)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
confusion_matrix_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
print(f"✅ Confusion Matrix saved to: {confusion_matrix_path}")
plt.close(fig_cm)

# -------------------------------
# 2. ROC Curve 저장 (이진 분류만)
# -------------------------------
if NUM_CLASSES == 2:
    all_probs = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Get Probs"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # NG 클래스 확률
            all_probs.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)  # NG가 양성 클래스라고 가정
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)
    plt.tight_layout()
    roc_curve_path = os.path.join(SAVE_DIR, "roc_curve.png")
    plt.savefig(roc_curve_path)
    print(f"✅ ROC Curve saved to: {roc_curve_path}")
    plt.close(fig_roc)