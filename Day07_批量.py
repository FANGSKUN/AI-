# -*- coding: utf-8 -*-
r"""
Day07 任务3练习：给线性回归加上 DataLoader 批量加载
====================================================
对比 Day07_线性回归.py：那次一次喂全部数据；
这次用 TensorDataset + DataLoader 分批喂。

把标了 TODO 的地方补全，然后运行 Day07_批量_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day07_批量_自检.py
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============ 1. 生成数据（和 Day 7 一样）============

def make_data(n=100, seed=0):
    """生成 y = 3x + 2 + 噪声，返回 (n,1) 的 X 和 y。"""
    torch.manual_seed(seed)
    X = torch.rand(n, 1)
    y = 3 * X + 2 + 0.1 * torch.randn(n, 1)
    return X, y


# ============ 2. 训练（TODO，核心）============

def train(X, y, lr=0.5, epochs=200, batch_size=32):
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # TODO 1：把 X、y 打包成 Dataset，再套上 DataLoader
    #   dataset = TensorDataset(X, y)
    #   loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        total, n = 0.0, 0
        # TODO 2：内层循环，一批批取数据，批内做标准 4 步
        for xb, yb in loader:
              optimizer.zero_grad()
              y_pre = model(xb)
              loss = loss_fn(y_pre, yb)
              loss.backward()
              optimizer.step()
              total += loss.item() * len(xb)
              n += len(xb)

        if n > 0:
            losses.append(total / n)        # 记录这个 epoch 的平均 loss

    return model, losses


# ============ 3. 主程序 ============

def main():
    X, y = make_data(200)
    model, losses = train(X, y)

    w = model.weight.item()
    b = model.bias.item()
    print(f"学到的 w = {w:.3f}  b = {b:.3f}   (真实 3 和 2)")
    print(f"最终 loss = {losses[-1]:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, s=10, label="数据")
    xs = torch.tensor([[0.0], [1.0]])
    with torch.no_grad():
        ys = model(xs)
    plt.plot(xs, ys, color="red", label="拟合直线")
    plt.legend()
    plt.title("线性回归拟合 (带 DataLoader)")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("损失下降曲线")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
