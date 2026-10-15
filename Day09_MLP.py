# -*- coding: utf-8 -*-
r"""
Day09 项目：多层感知机 MLP——用 XOR 理解"为什么需要非线性"
========================================================
核心问题：XOR 不是线性可分的——单层 Linear 无论怎么调 w、b，
都画不出一条直线把两类分开（准确率会卡在 50% 左右）。

解法：加隐藏层 + 非线性激活 ReLU，让模型能画"折线/曲线"边界。

对比：
    Day 8 逻辑回归：Linear(2,1) -> Sigmoid                （只能画直线）
    Day 9 MLP：       Linear(2,h) -> ReLU -> Linear(h,1) -> Sigmoid（能画非直线）

把标了 TODO 的地方补全，然后运行 Day09_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day09_自检.py
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============ 1. 生成 XOR 数据 ============

def make_xor(n_per_corner=25, seed=0):
    """生成 XOR 四角数据，返回 (X, y)。

    四角（加一点噪声）：
        (-1,-1) -> 0   (1,1) -> 0   （同号 = 0 类）
        (-1,1)  -> 1   (1,-1) -> 1  （异号 = 1 类）
    X 形状 (4*n_per_corner, 2)，y 形状 (4*n_per_corner, 1)。
    """
    torch.manual_seed(seed)
    # TODO 1：四个角的点（乘 0.3 缩小噪声），按 0,0,1,1 拼标签
    X0 = torch.randn(n_per_corner, 2) * 0.3 + torch.tensor([-1.0, -1.0])
    X1 = torch.randn(n_per_corner, 2) * 0.3 + torch.tensor([1.0, 1.0])
    X2 = torch.randn(n_per_corner, 2) * 0.3 + torch.tensor([-1.0, 1.0])
    X3 = torch.randn(n_per_corner, 2) * 0.3 + torch.tensor([1.0, -1.0])
    X = torch.cat([X0, X1, X2, X3], dim=0)
    y = torch.cat([torch.zeros(2 * n_per_corner, 1), torch.ones(2 * n_per_corner, 1)], dim=0)
    return X, y


# ============ 2. 单层逻辑回归（复习 Day 8，会在 XOR 上失败）============

def train_linear(X, y, lr=0.5, epochs=1000):
    """单层线性模型，和 Day 8 一样。在 XOR 上准确率会卡在 50% 左右。"""
    model = nn.Linear(2, 1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        p = torch.sigmoid(model(X))
        loss = loss_fn(p, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


# ============ 3. MLP（TODO，核心）============

def train_mlp(X, y, lr=0.1, epochs=1000, hidden=4):
    """两层网络：加隐藏层 + ReLU，返回 (model, losses)。"""
    # TODO 2：用 nn.Sequential 搭网络：Linear(2,hidden) -> ReLU -> Linear(hidden,1)
    #   提示：nn.ReLU() 是激活函数，作用是 max(0, x)
    #   model = nn.Sequential(
    #       nn.Linear(2, hidden),
    #       nn.ReLU(),
    #       nn.Linear(hidden, 1),
    #   )
    model = nn.Sequential(
        nn.Linear(2, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        p = torch.sigmoid(model(X))    # model 是 Sequential，直接 model(X) 前向
        loss = loss_fn(p, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


# ============ 4. 预测与评估（和 Day 8 一样）============

def predict(model, X):
    with torch.no_grad():
        p = torch.sigmoid(model(X))
        return (p >= 0.5).float()


def accuracy(model, X, y):
    pred = predict(model, X)
    return (pred == y).float().mean().item()


# ============ 5. 画决策边界 ============

def plot_boundary(model, X, y, ax, title):
    with torch.no_grad():
        xs = torch.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
        ys = torch.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        zz = torch.sigmoid(model(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, cmap="bwr")
    ax.contour(xx, yy, zz, levels=[0.5], colors="red")
    ax.scatter(X[:, 0][y[:, 0] == 0], X[:, 1][y[:, 0] == 0], c="blue", label="0 类", s=20)
    ax.scatter(X[:, 0][y[:, 0] == 1], X[:, 1][y[:, 0] == 1], c="orange", label="1 类", s=20)
    ax.set_title(title)
    ax.legend()


def main():
    X, y = make_xor()

    lin_model, lin_losses = train_linear(X, y)
    mlp_model, mlp_losses = train_mlp(X, y)

    print(f"线性模型准确率 = {accuracy(lin_model, X, y):.3f}  （应约 0.5，学不动）")
    print(f"MLP 准确率     = {accuracy(mlp_model, X, y):.3f}  （应接近 1）")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_boundary(lin_model, X, y, axes[0, 0], "线性模型：画不出非直线（失败）")
    plot_boundary(mlp_model, X, y, axes[0, 1], "MLP：画出非直线边界（成功）")
    axes[1, 0].plot(lin_losses)
    axes[1, 0].set_title("线性模型 loss（卡住下不去）")
    axes[1, 1].plot(mlp_losses)
    axes[1, 1].set_title("MLP loss（降下去）")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
