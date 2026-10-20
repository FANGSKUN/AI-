# -*- coding: utf-8 -*-
"""
Day15 扩展：用训练好的 CNN 预测你自己的照片
用法：
    conda activate ai
    python Day15_预测图片.py <图片路径>

会先加载已保存的权重（Day15_model.pth），若没有则训练 5 轮并保存，
然后把你的照片缩成 32×32 喂给模型，输出 Top3 预测类别和概率。
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(2048, 10),
    )


def load_or_train():
    model = build_cnn().to(device)
    path = "Day15_model.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print("已加载已保存的权重 Day15_model.pth")
        return model
    # 训练 5 轮并保存
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print(f"训练 epoch {epoch+1}/5 完成")
    torch.save(model.state_dict(), path)
    print(f"权重已保存到 {path}")
    return model


def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0).to(device)  # (1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top3 = probs.argsort(descending=True)[:3]
    print(f"\n图片：{image_path}")
    for rank, idx in enumerate(top3, 1):
        print(f"  Top{rank}: {CLASSES[idx]}  ({probs[idx]:.1%})")


if __name__ == "__main__":
    model = load_or_train()
    if len(sys.argv) > 1:
        predict(model, sys.argv[1])
    else:
        print("用法：python Day15_预测图片.py <图片路径>")
