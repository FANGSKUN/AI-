# -*- coding: utf-8 -*-
r"""
Day07 项目：用 nn + optim 重构线性回归
========================================
对比 Day06：
    - Day06 手建参数 w、b，手动写 backward、更新、清零
    - Day07 用 nn.Linear / nn.MSELoss / optim.SGD，
      训练循环变成标准 4 步

把标了 TODO 的地方补全，然后运行 Day07_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day07_自检.py
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============ 1. 生成数据（和 Day 6 一样）============

def make_data(n=100, seed=0):
    """生成 y = 3x + 2 + 噪声，返回 (n,1) 的 X 和 y。"""
    torch.manual_seed(seed)
    X = torch.rand(n, 1)
    y = 3 * X + 2 + 0.1 * torch.randn(n, 1)
    return X, y


# ============ 2. 训练（TODO，核心）============

def train(X, y, lr=0.5, epochs=200):
    """标准训练循环，返回 (model, losses)。

    对比 Day06 的三步，现在变成标准 4 步：
        1. optimizer.zero_grad()   # 清零所有参数梯度
        2. loss.backward()         # 反向传播
        3. optimizer.step()        # 更新所有参数（内部已 no_grad）
       （loss 计算夹在 1 和 2 之间）
    """
    model = nn.Linear(1, 1)                                # 参数 w、b 自动建好
    loss_fn = nn.MSELoss()                                 # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 优化器接管参数
    losses = []

    for epoch in range(epochs):
        # TODO 1：清零梯度（一次清 model 的所有参数）
        #   optimizer.zero_grad()
        optimizer.zero_grad()
        y_pre = model(X)                                  # 前向

        # TODO 2：算损失
        #   loss = loss_fn(y_pre, y)
        loss = loss_fn(y_pre, y)
        # TODO 3：反向传播
        #   loss.backward()
        loss.backward()
        # TODO 4：更新参数
        #   optimizer.step()
        optimizer.step()
        losses.append(loss.item())                         # .item() 取出数字

    return model, losses


# ============ 3. 主程序 ============

def main():
    X, y = make_data()
    model, losses = train(X, y)

    w = model.weight.item()
    b = model.bias.item()
    print(f"学到的 w = {w:.3f}  b = {b:.3f}   (真实 3 和 2)")
    print(f"初始 loss = {losses[0]:.4f}  最终 loss = {losses[-1]:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, s=10, label="数据")
    xs = torch.tensor([[0.0], [1.0]])
    with torch.no_grad():
        ys = model(xs)
    plt.plot(xs, ys, color="red", label="拟合直线")
    plt.legend()
    plt.title("线性回归拟合 (nn+optim)")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("损失下降曲线")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
