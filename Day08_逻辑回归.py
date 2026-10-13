# -*- coding: utf-8 -*-
r"""
Day08 项目：逻辑回归（二分类）
====================================
对比 Day07 线性回归：
    - 回归：输出一个实数，用 MSE 损失
    - 分类：输出一个概率，用 BCE 损失

流程：X -> Linear(z) -> Sigmoid(p) -> 阈值 0.5 -> 预测 0/1

把标了 TODO 的地方补全，然后运行 Day08_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day08_自检.py
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============ 1. 生成二分类数据 ============

def make_data(n_per_class=100, seed=0):
    """生成两团 2D 数据：0 类在左下，1 类在右上，返回 (X, y)。

    X 形状 (2*n_per_class, 2)，y 形状 (2*n_per_class, 1)。
    """
    torch.manual_seed(seed)
    # TODO 1：生成两团高斯点 + 对应标签，再拼起来
    #   X0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, -2.0])   # 0 类：左下
    #   X1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 2.0])     # 1 类：右上
    #   X = torch.cat([X0, X1], dim=0)
    #   y = torch.cat([torch.zeros(n_per_class, 1), torch.ones(n_per_class, 1)], dim=0)
    X0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, -2.0])
    X1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 2.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n_per_class, 1), torch.ones(n_per_class, 1)], dim=0)
    return X, y


# ============ 2. 训练（TODO，核心）============

def train(X, y, lr=0.1, epochs=300):
    """用 Linear + Sigmoid + BCELoss 训练，返回 (model, losses)。

    注意和 Day07 的区别：模型输出 z 是 logits（任意实数），
    必须先过 sigmoid 变成概率，再喂给 BCELoss。
    """
    model = nn.Linear(2, 1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(X)                       # 前向：线性输出 logits
        # TODO 2：把 logits 压成概率
        #   p = torch.sigmoid(z)
        p = torch.sigmoid(z)
        # TODO 3：算 BCE 损失（注意喂的是 p，不是 z）
        #   loss = loss_fn(p, y)
        loss = loss_fn(p, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


# ============ 3. 预测与评估（TODO）============

def predict(model, X):
    """返回预测标签 (n,1)，元素为 0 或 1。"""
    with torch.no_grad():
        p = torch.sigmoid(model(X))
        # TODO 4：p >= 0.5 判为 1 类，否则 0 类（转成 float）
        #   return (p >= 0.5).float()
        return (p >= 0.5).float()


def accuracy(model, X, y):
    """准确率：预测正确的样本占比。"""
    # TODO 5：比较 predict 结果和真实标签 y，求平均
    #   pred = predict(model, X)
    #   return (pred == y).float().mean().item()
    pred = predict(model, X)
    return (pred == y).float().mean().item()
    pass


# ============ 4. 主程序（已写好，跑通后看决策边界）============

def plot_boundary(model, X, y):
    """画散点 + 决策边界（p=0.5 的那条线）。"""
    with torch.no_grad():
        xs = torch.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
        ys = torch.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        p = torch.sigmoid(model(grid))
        zz = p.reshape(xx.shape)

    plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, cmap="bwr")
    plt.contour(xx, yy, zz, levels=[0.5], colors="red")   # 决策边界
    plt.scatter(X[:, 0][y[:, 0] == 0], X[:, 1][y[:, 0] == 0], c="blue", label="0 类", s=20)
    plt.scatter(X[:, 0][y[:, 0] == 1], X[:, 1][y[:, 0] == 1], c="orange", label="1 类", s=20)
    plt.legend()
    plt.title("逻辑回归决策边界")
    plt.tight_layout()


def main():
    X, y = make_data()
    model, losses = train(X, y)

    acc = accuracy(model, X, y)
    w = model.weight.detach().numpy().reshape(-1)
    b = model.bias.item()
    print(f"权重 w = {w}   偏置 b = {b:.3f}")
    print(f"训练准确率 = {acc:.4f}")
    print(f"初始 loss = {losses[0]:.4f}   最终 loss = {losses[-1]:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_boundary(model, X, y)

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("损失下降曲线")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
