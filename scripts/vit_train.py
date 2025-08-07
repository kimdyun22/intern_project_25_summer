import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
from tqdm import tqdm

# 경로 설정 (윈도우 -> 리눅스 컨테이너 경로로 자동 변환)
DATA_DIR = "/workspace/dataset_split"
BATCH_SIZE = 32
NUM_EPOCHS = 20
IMAGE_SIZE = 224
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 모델 불러오기
model = create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.to(DEVICE)

# 손실 함수, 옵티마이저, 스케줄러
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# 학습 함수
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in tqdm(train_loader, desc="Train"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / len(train_dataset)
    return epoch_loss, epoch_acc

# 검증 함수
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Val  "):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = correct / len(val_dataset)

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))
    return epoch_loss, epoch_acc

# 학습 루프
for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    scheduler.step()

    print(f"✅ Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"🔎  Val  Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

# 모델 저장
torch.save(model.state_dict(), "vit_brake_classifier.pth")
print("💾 모델 저장 완료: vit_brake_classifier.pth")
