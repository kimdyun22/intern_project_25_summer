import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ======== 설정 ========
DATA_DIR = "/workspace/dataset_split/train/OK"  # OK 데이터 경로
SAVE_PATH = "/workspace/autoencoder_model.pth"  # 모델 저장 경로
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== 데이터셋 정의 ========
class OKDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = OKDataset(DATA_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ======== 오토인코더 정의 ========
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # [B, 64, 112, 112]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B, 128, 56, 56]
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # [B, 256, 28, 28]
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # [B, 512, 14, 14]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [B, 256, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B, 128, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # [B, 64, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # [B, 3, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ======== 학습 ========
model = AutoEncoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("🚀 Start training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"📉 Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.6f}")

# ======== 저장 ========
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ 모델이 저장되었습니다 → {SAVE_PATH}")
