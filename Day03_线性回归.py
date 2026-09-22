# -*- coding: utf-8 -*-
r"""
Day03 项目：纯 NumPy 手写线性回归
==================================
目标：从零实现"用梯度下降拟合 y = 3x + 2"。

把下面标了 TODO 的地方补全，然后运行 Day03_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day03_自检.py
"""
import numpy as np
import matplotlib.pyplot as plt


# ============ 1. 生成数据 ============

def make_data(n=100, seed=0):
    """生成 n 个样本，真实关系 y = 3x + 2 + 噪声。
    返回 X (n,1), y (n,1)。
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 1))                 # (n,1)，[0,1) 均匀
    noise = rng.normal(0, 0.1, (n, 1))     # 均值0、标准差0.1 的噪声
    y = 3 * X + 2 + noise
    return X, y


# ============ 2. 模型 ============

def predict(X, w, b):
    """线性模型：y_pred = X @ w + b。"""
    return X @ w + b


# ============ 3. 损失函数（TODO）============

def mse_loss(y_pred, y):
    """均方误差：MSE = mean((y_pred - y) ** 2)。
    返回一个标量。
    """
    # TODO: 实现 MSE
    Mse = np.mean((y_pred - y)**2)
    return Mse


# ============ 4. 梯度（TODO，核心）============

def gradients(X, y, w, b):
    """返回损失对 w 和 b 的梯度 (dw, db)。

    推导提示：
        y_pred = X @ w + b
        err = y_pred - y
        MSE = mean(err^2)
        对 w 求导：dw = 2 * mean(err * X)
        对 b 求导：db = 2 * mean(err)
    """
    # TODO: 实现，返回 (dw, db)
    # 注意：dw 的形状要能直接和 w(1,1) 相减
    err = predict(X, w, b) - y
    dw = 2 * np.mean(err * X)
    db = 2 * np.mean(err)
    return dw, db


# ============ 5. 训练（TODO）============

def train(X, y, lr=0.5, epochs=200):
    """梯度下降训练，返回 (w, b, losses)。
    初始 w = np.zeros((1,1)), b = 0.0。
    losses 记录每一轮的 loss，用于画曲线。
    """
    w = np.zeros((1, 1))
    b = 0.0
    losses = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

        dw, db = gradients(X, y, w, b)

        # TODO: 用梯度下降更新 w 和 b
        # 公式：w = w - lr * dw ; b = b - lr * db
        w = w - lr * dw
        b = b - lr * db

    return w, b, losses


# ============ 6. 主程序（画图）============

def main():
    X, y = make_data()
    w, b, losses = train(X, y)

    print(f"学到的 w = {w.ravel()[0]:.3f}  b = {b:.3f}   (真实值 w=3, b=2)")
    print(f"初始 loss = {losses[0]:.4f}  最终 loss = {losses[-1]:.4f}")

    plt.figure(figsize=(10, 4))

    # 左图：数据散点 + 拟合直线
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, s=10, label="数据")
    xs = np.array([[0.0], [1.0]])
    plt.plot(xs, predict(xs, w, b), color="red", label="拟合直线")
    plt.legend()
    plt.title("线性回归拟合")

    # 右图：loss 下降曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("损失下降曲线")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
