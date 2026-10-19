# -*- coding: utf-8 -*-
r"""
Day13 项目：MNIST 手写数字识别——第一次做"多分类"
================================================
之前全是二分类（0 或 1：输出 1 个神经元 + sigmoid + BCE）。
今天上真实数据集 MNIST：28×28 灰度手写数字，0~9 共 10 类。

多分类相对二分类的三个关键变化：
  1. 输出层：1 个神经元 → 10 个神经元（每个数字一个"得分" logit）
  2. 损失函数：BCELoss → CrossEntropyLoss（内部已含 softmax，标签用整数 0~9）
  3. 预测：sigmoid 后 >0.5 → 直接 argmax 取"得分最高"的那个类

把标了 TODO 的地方补全，然后运行 Day13_自检.py 检查。

运行：
    conda activate ai
    python Day13_自检.py
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============ 1. 加载 MNIST ============

def load_mnist(batch_size=64):
    """加载 MNIST，返回 (train_loader, test_loader)。

    首次运行会联网下载（约 12MB），已缓存在 ./data/ 里。
    """
    transform = transforms.ToTensor()  # PIL 图 -> (1,28,28) 张量，像素值 0~1
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============ 2. 多分类模型 ============

def build_model(hidden=128):
    """构建多分类 MLP：输入 784（28×28 展平），输出 10（0~9 十类）。"""
    model = nn.Sequential(
        nn.Flatten(),                 # (B,1,28,28) -> (B,784)
        nn.Linear(784, hidden),
        nn.ReLU(),
        # TODO 1：最后一个 Linear 的输出改成 10（多分类，每个类一个 logit）。
        #   二分类是 nn.Linear(hidden, 1)；多分类改成 nn.Linear(hidden, 10)。
        nn.Linear(hidden, 10),         # ← 改这里
    )
    return model


def build_loss():
    """返回多分类损失函数。"""
    # TODO 2：把二分类的 BCELoss 换成多分类的 CrossEntropyLoss。
    #   注意：CrossEntropyLoss 内部已经做了 softmax，所以模型输出 logits 就行、
    #   不要再套 sigmoid；标签直接是整数 0~9（不是 one-hot）。
    return nn.CrossEntropyLoss()           # ← 改这里


def predict(model, x):
    """对一批图像 x (B,1,28,28) 预测类别，返回整数标签张量 (B,)。

    多分类的预测 = 从 10 个 logits 里取"得分最高"的那个类的下标。
    """
    # TODO 3：返回 argmax。
    #   提示：model(x) 得到 (B,10) 的 logits，用 .argmax(dim=1) 取每行最大值下标。
    logits = model(x)
    return logits.argmax(dim=1)


# ============ 3. 训练 / 评估 ============

def train_epoch(model, loader, opt, loss_fn):
    """训练一轮，返回 (平均损失, 训练准确率)。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        opt.zero_grad()
        logits = model(x)              # (B,10)，多分类不要 sigmoid
        loss = loss_fn(logits, y)      # y 是整数标签 0~9
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def evaluate(model, loader, loss_fn):
    """在测试集上评估，返回 (平均损失, 准确率)。"""
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

    model = build_model(hidden=128)
    loss_fn = build_loss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        print(f"epoch {epoch+1}: train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f}  "
              f"test_loss {te_loss:.4f}  test_acc {te_acc:.4f}")

    # 打印几个预测示例
    dataiter = iter(test_loader)
    x, y = next(dataiter)
    pred = predict(model, x)
    print(f"\n前 10 个样本：真实 {y[:10].tolist()}")
    print(f"            预测 {pred[:10].tolist()}")


if __name__ == "__main__":
    main()
