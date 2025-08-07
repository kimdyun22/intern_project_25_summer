import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

# ========== 1. 설정 =========
data_dir = "/workspace/gan_input" # NG 이미지만 있는 폴더 하위 구조: gan_input/NG/*.bmp
save_dir = "/workspace/gan_generated" 
os.makedirs(save_dir, exist_ok=True)

image_size = 64
batch_size = 64
nz = 100  # 생성기 입력 noise 벡터 크기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. 데이터셋 ==========
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ========== 3. 모델 정의 ==========
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

# ========== 4. 학습 ==========
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 100
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(tqdm(dataloader)):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # 진짜 이미지 label: 1, 가짜 이미지 label: 0
        real_labels = torch.full((batch_size, 1), 1.0, device=device)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)

        # === 1. Discriminator 학습 ===
        optimizer_d.zero_grad()
        output_real = discriminator(real_images)
        loss_d_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_d_fake = criterion(output_fake, fake_labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # === 2. Generator 학습 ===
        optimizer_g.zero_grad()
        output = discriminator(fake_images)
        loss_g = criterion(output, real_labels)  # Generator는 진짜처럼 보이도록
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

    # 중간 결과 저장
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        grid = make_grid(fake, padding=2, normalize=True)
        save_image(grid, os.path.join(save_dir, f"generated_epoch_{epoch+1}.png"))

# ========== 5. 최종 가짜 이미지 저장 ==========
generator.eval()
with torch.no_grad():
    for i in range(200):
        z = torch.randn(1, nz, 1, 1, device=device)
        fake_img = generator(z).detach().cpu()
        save_image(fake_img, os.path.join(save_dir, f"fake_ng_{i:04}.bmp"), normalize=True)
