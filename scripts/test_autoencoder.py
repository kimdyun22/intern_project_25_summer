import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ======== 설정 ========
MODEL_PATH = "/workspace/autoencoder_model.pth"
TEST_DIR = "/workspace/dataset_split/test"
IMG_SIZE = 224
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECON_THRESHOLD = 0.01  # 재구성 오차 임계값 (조절 필요)

# ======== AutoEncoder 정의 ========
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 4, 2, 1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ======== 데이터셋 정의 ========
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for cls in ["OK", "NG"]:
            cls_path = os.path.join(root_dir, cls)
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith(".bmp")]
            self.samples.extend([(f, 0 if cls == "OK" else 1) for f in files])
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label, path


# ======== 전처리 ========
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = TestDataset(TEST_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======== 모델 로드 ========
model = AutoEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======== 평가 ========
y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels, paths in dataloader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = torch.nn.functional.mse_loss(outputs, imgs, reduction='mean')

        pred = 1 if loss.item() > RECON_THRESHOLD else 0  # 1: NG, 0: OK
        y_pred.append(pred)
        y_true.append(labels.item())

# ======== 결과 출력 ========
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=["OK", "NG"]))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
