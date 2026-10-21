# -*- coding: utf-8 -*-
r"""
Day16 项目：数据增强（Data Augmentation）
========================================
Day15 的三层 CNN 在 CIFAR-10 上 test_acc 只有 0.70。
今天**不加参数、不改网络结构**，只靠「数据增强」，就把 test_acc 从 0.70 提到 0.76+（继续训练 20~30 轮可冲 0.80）。

核心思想（用一句话记住）：
  模型学的是「训练集的像素分布规律」，不是「猫这个概念」。
  CIFAR-10 里的猫有无数种姿势（侧躺/奔跑/被挡一半/从背面看）。
  如果模型只见过「标准姿势」，测试时遇到「侧躺的猫」就认不出。

  数据增强 = 训练时把每张图**随机折腾一下**（翻转、裁剪），
  让模型见多识广，被迫学「本质特征」而不是「某一张具体的图」。

铁律（今天的重中之重）：
  只增强**训练集**，测试集保持原样。
  类比：考试要考「原题」，不能把卷子也变花样，否则分数是假的。

今天 4 个 TODO 全在 transforms 上（torchvision 现成的增强函数）。

运行：
    conda activate ai
    python Day16_自检.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


# ============ 1. 加载数据（今天的核心：训练集加增强，测试集不加） ============

def load_cifar(batch_size=64):
    # 训练集：加数据增强，让模型「见多识广」
    transform_train = transforms.Compose([
        # TODO 1：随机水平翻转（左右镜像，让猫朝左朝右都能认）
        #   提示：torchvision 里叫 RandomHorizontalFlip
        transforms.RandomHorizontalFlip(),

        # TODO 2：随机裁剪（先四周补 4 圈边，再随机裁回 32×32，让物体位置/大小变化也能认）
        #   提示：函数名 RandomCrop，第一个参数是裁剪尺寸 32，第二个参数 padding=4
        transforms.RandomCrop(32, padding=4),

        transforms.ToTensor(),
    ])
    # 测试集：不增强，保持原样（要「原题」评估真本事）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # TODO 3：训练集该用哪个 transform？（填变量名 transform_train / transform_test）
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True,
                                transform=transform_train)
    # TODO 4：测试集该用哪个 transform？
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True,
                               transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============ 2. 三层 CNN（和 Day15 标准版一样，32→64→128，今天不改它） ============

def build_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),   # 32×32 → 16×16

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),   # 16×16 → 8×8

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),   # 8×8 → 4×4

        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 10),
    )
    return model


# ============ 3. 训练 / 评估（和 Day15 一样） ============

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


# ============ 4. 主程序：训练 5 轮，对比 Day15 无增强的 0.70 ============

def main():
    train_loader, test_loader = load_cifar(batch_size=64)

    model = build_cnn().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    print("数据增强版：训练 15 轮")
    print("注意：增强让训练变慢，前几轮 test_acc 会低于无增强版；")
    print("但训练足够久后（15 轮），增强版会反超——这就是「防过拟合」的威力。\n")
    for epoch in range(15):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        print(f"epoch {epoch+1}: train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f}  "
              f"test_loss {te_loss:.4f}  test_acc {te_acc:.4f}")


if __name__ == "__main__":
    main()
