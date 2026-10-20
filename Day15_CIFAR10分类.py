# -*- coding: utf-8 -*-
r"""
Day15 项目：CIFAR-10 彩色图像分类（3 通道 CNN，进阶）
================================================
Day14 用 CNN 做 MNIST（灰度图 1 通道），test_acc 0.988，已经饱和。
今天升级到 CIFAR-10：32×32 的**彩色图**（RGB 3 通道），10 类真实物体
（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车）。

今天的两个新东西：
  1. **3 通道**：彩色图是 R、G、B 三张图叠在一起，in_channels 从 1 变 3。
     第一层每个卷积核要同时"看"R、G、B 三张图（所以是 3×3×3 的权重）。
  2. **更深的网络**：CIFAR-10 比 MNIST 难得多（物体复杂、有颜色、有背景），
     两层卷积不够，要三层（32→64→128 通道），才能逐层抽象出"高级特征"。

维度变化（自己脑内跟着算）：
  32×32 → Conv(3→32) → Pool → 16×16 → Conv(32→64) → Pool → 8×8
        → Conv(64→128) → Pool → 4×4 → Flatten → 128×4×4 = ?

运行：
    conda activate ai
    python Day15_自检.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


# ============ 1. 加载 CIFAR-10（第一次运行会自动下载，约 170MB） ============

def load_cifar(batch_size=64):
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============ 2. 三层 CNN（今天的核心，4 个 TODO） ============

def build_cnn():
    """三层 CNN：卷积→池化 重复三次 → 展平 → 全连接。

    输入 (B,3,32,32) 逐层变化：
      Conv2d(3,32,3,padding=1)  → (B,32,32,32)  尺寸不变
      MaxPool2d(2)              → (B,32,16,16)  减半
      Conv2d(32,64,3,padding=1) → (B,64,16,16)
      MaxPool2d(2)              → (B,64,8,8)    减半
      Conv2d(64,128,3,padding=1)→ (B,128,8,8)
      MaxPool2d(2)              → (B,128,4,4)   减半
      Flatten                   → (B, 128*4*4 = ?)
      Linear(?, 10)             → (B, 10)
    """
    model = nn.Sequential(
        # TODO 1：第一个卷积层，补全 in_channels。
        #   CIFAR-10 是彩色图（RGB 3 通道），不像 MNIST 灰度图只有 1 通道。
        #   每个卷积核要同时看 R、G、B 三张图，所以 in_channels = ?
        nn.Conv2d(3, 64, kernel_size=3, padding=1),

        nn.ReLU(),
        nn.MaxPool2d(2),   # 32×32 → 16×16

        # TODO 2：第二个卷积层，补全 in_channels（接上一层输出 32 通道）。
        nn.Conv2d( 64, 128,kernel_size=3, padding=1),

        nn.ReLU(),
        nn.MaxPool2d(2),   # 16×16 → 8×8

        # TODO 3：第三个卷积层，补全 in_channels（接上一层输出 64 通道）。
        nn.Conv2d(128, 256,kernel_size=3, padding=1),

        nn.ReLU(),
        nn.MaxPool2d(2),   # 8×8 → 4×4

        nn.Flatten(),
        # TODO 4：全连接层，补全 in_features。
        #   自己算展平后的特征数 = 通道数 128 × 高 4 × 宽 4 = ?
        nn.Linear(4096, 10),
    )
    return model


# ============ 3. 训练 / 评估（和 Day14 一样，只是加了 device） ============

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


# ============ 4. 主程序 ============

def main():
    train_loader, test_loader = load_cifar(batch_size=64)

    model = build_cnn().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\n总参数量：{n_params:,}（对比 Day14 的 CNN 约 2 万，CIFAR-10 复杂得多）\n")

    for epoch in range(5):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        print(f"epoch {epoch+1}: train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f}  "
              f"test_loss {te_loss:.4f}  test_acc {te_acc:.4f}")


if __name__ == "__main__":
    main()
