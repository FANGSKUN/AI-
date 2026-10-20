# -*- coding: utf-8 -*-
r"""
Day14 项目：CNN 卷积神经网络——从 MLP 到 CNN
================================================
Day13 用 MLP（Flatten + Linear + ReLU）做 MNIST，test_acc 0.968。
但 MLP 有个先天缺陷：nn.Flatten 把 28×28 的图摊成 784 个散落的数，
「像素之间的空间位置关系」全丢了。

比如数字「7」是一横 + 一竖 + 一斜，这个笔画结构在 Flatten 眼里
只是「第 3 个像素亮、第 27 个像素亮…」的散点，它不知道这些像素
在图上挨在一起、构成一条竖线。

CNN（卷积神经网络）就是来解决这个的：它用一个小窗口（卷积核 kernel）
在图上「滑动」，每次只看一小块区域，从而保留「谁在谁旁边」的空间结构。

今天把 MLP 换成 CNN，对比两者的结构和效果。

先理解三个新组件（填 TODO 前必读）：
  1. 卷积层 nn.Conv2d(in_channels, out_channels, kernel_size, padding)
     - in_channels ：输入通道数（灰度图 = 1）
     - out_channels：输出通道数 = 卷积核的个数（每个卷积核学一种「特征」，
                     比如一个学「横线」、一个学「竖线」）
     - kernel_size ：卷积核大小（3 = 3×3 的小窗口）
     - padding      ：在图像边缘补一圈 0，让卷积后尺寸不变（1 = 补一圈）
  2. 池化层 nn.MaxPool2d(2)：把特征图缩小一半（2×2 里取最大），
     相当于「拍照后压缩」，保留最显著特征、减少计算量
  3. 最后还是要 Flatten + Linear：卷积提取完特征后，摊平喂给全连接层做分类

运行：
    conda activate ai
    python Day14_自检.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============ 1. 加载 MNIST（同 Day13） ============

def load_mnist(batch_size=64):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============ 2. CNN 模型（今天的核心，4 个 TODO） ============

def build_cnn():
    """构建 CNN：卷积 → 池化 → 卷积 → 池化 → 展平 → 全连接。

    输入 (B,1,28,28) 逐层变化（自己脑内跟着算一遍尺寸）：
      Conv2d(1,16,3,padding=1) → (B,16,28,28)   padding=1 尺寸不变
      ReLU
      MaxPool2d(2)             → (B,16,14,14)   尺寸减半
      Conv2d(16,32,3,padding=1)→ (B,32,14,14)   尺寸不变
      ReLU
      MaxPool2d(2)             → (B,32,7,7)     尺寸再减半
      Flatten                  → (B, 32*7*7 = ?) 展平
      Linear(?, 10)            → (B, 10)        分类
    """
    model = nn.Sequential(
        # TODO 1：第一个卷积层，补全 in_channels。
        #   输入是灰度图 = 几个通道？（提示：Day13 讲过灰度图是 (1,28,28)）
        nn.Conv2d(1, 16, kernel_size=3, padding=1),

        nn.ReLU(),
        # TODO 2：池化层，补全参数。2×2 里取最大、尺寸减半。
        nn.MaxPool2d(2),

        # TODO 3：第二个卷积层，补全 in_channels。
        #   这一层输入的是上一层的输出（16 个特征），所以 in_channels = 上一层的 out_channels。
        nn.Conv2d(16, 32, kernel_size=3, padding=1),

        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        # TODO 4：全连接层，补全 in_features。
        #   自己算展平后的特征数：
        #     28×28 → 卷积(padding=1 不变) → 池化(÷2) → 14×14
        #           → 卷积(不变) → 池化(÷2) → 7×7
        #   展平 = 通道数 32 × 高 7 × 宽 7 = ?
        nn.Linear(1568, 10),
    )
    return model


# ============ 3. 训练 / 评估（同 Day13，不用改） ============

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
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
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


# ============ 4. 主程序 ============

def main():
    train_loader, test_loader = load_mnist(batch_size=64)

    model = build_cnn()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构 + 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\n总参数量：{n_params:,}（对比 Day13 的 MLP 约 10 万，CNN 少得多）\n")

    for epoch in range(5):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        print(f"epoch {epoch+1}: train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f}  "
              f"test_loss {te_loss:.4f}  test_acc {te_acc:.4f}")


if __name__ == "__main__":
    main()
