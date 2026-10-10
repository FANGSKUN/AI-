# -*- coding: utf-8 -*-
r"""
Day06 项目：用 PyTorch 重写线性回归
======================================
对比 Day 3 的手写 NumPy 版：那时你要自己推梯度 dw、db，
现在 autograd 自动算，你只需要写 forward 和 loss。

把标了 TODO 的地方补全，然后运行 Day06_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day06_自检.py
"""
import matplotlib.pyplot as plt
import torch


# ============ 1. 生成数据（和 Day 3 一样）============

def make_data(n=100, seed=0):
    """生成 y = 3x + 2 + 噪声，返回 (n,1) 的 X 和 y。"""
    torch.manual_seed(seed)
    X = torch.rand(n, 1)
    y = 3 * X + 2 + 0.1 * torch.randn(n, 1)
    return X, y


# ============ 2. 训练（TODO，核心）============

def train(X, y, lr=0.5, epochs=200):
    """梯度下降训练，返回 (w, b, losses)。

    和 Day 3 的唯一区别：不用手写 dw、db 公式了，
    用 autograd 自动算。三步：
        1. loss.backward()               # 反向传播，自动把梯度存进 w.grad、b.grad
        2. 更新 w、b                      # w = w - lr * w.grad
        3. 清零梯度                       # 否则下一轮梯度会累加
    """
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []

    for epoch in range(epochs):
        y_pred = X @ w + b                          # 前向：线性模型
        loss = ((y_pred - y) ** 2).mean()           # MSE 损失（标量）

        # TODO 1：反向传播，自动求梯度
        #   loss.backward()
        loss.backward()
        # TODO 2：更新参数
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        # TODO 3：清零梯度（原地操作，末尾加下划线）
        #   w.grad.zero_()
        #   b.grad.zero_()
        w.grad.zero_()
        b.grad.zero_()
        losses.append(loss.item())                  # .item() 取出 Python 数字

    return w, b, losses


# ============ 3. 主程序 ============

def main():
    X, y = make_data()
    w, b, losses = train(X, y)

    print(f"学到的 w = {w.item():.3f}  b = {b.item():.3f}   (真实 3 和 2)")
    print(f"初始 loss = {losses[0]:.4f}  最终 loss = {losses[-1]:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, s=10, label="数据")
    xs = torch.tensor([[0.0], [1.0]])
    with torch.no_grad():
        ys = xs @ w + b
    plt.plot(xs, ys, color="red", label="拟合直线")
    plt.legend()
    plt.title("线性回归拟合 (PyTorch)")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("损失下降曲线")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
