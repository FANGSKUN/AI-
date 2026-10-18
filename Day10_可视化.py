# -*- coding: utf-8 -*-
"""
Day10 补充可视化：直观看"过拟合"的决策边界
==========================================
画 2x2 四张子图：
  上排 = 两团"有重叠"的数据（noise=1.0）
  下排 = 两团"完全可分"的数据（noise=0.5，离得更远）
  左列 = 小模型 (hidden=4)
  右列 = 大模型 (hidden=128)

看点：
  - 大模型在"重叠数据"上，边界会绕开每个训练点（弯弯绕绕），验证集出现分错的点
  - 大模型在"完全可分"数据上，边界老老实实待在缝里，验证集照样全对 → 过拟合"隐身"
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_data(n=400, seed=0, noise=1.0, c0=(-1.0, -1.0), c1=(1.0, 1.0)):
    torch.manual_seed(seed)
    n0 = n // 2
    X0 = torch.randn(n0, 2) * noise + torch.tensor(c0)
    X1 = torch.randn(n - n0, 2) * noise + torch.tensor(c1)
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n0, 1), torch.ones(n - n0, 1)], dim=0)
    return X, y


def split(X, y, train_ratio=0.25, seed=7):
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n)
    k = int(n * train_ratio)
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


def train_mlp(X_tr, y_tr, hidden=64, epochs=8000, lr=0.01):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(torch.sigmoid(model(X_tr)), y_tr)
        loss.backward()
        opt.step()
    return model


def proba(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(X)).squeeze()


def acc(model, X, y):
    p = proba(model, X)
    return ((p >= 0.5).float() == y.squeeze()).float().mean().item()


def draw(ax, model, X_tr, y_tr, X_va, y_va, title, lim=(-4.5, 4.5)):
    # 背景网格，算每个点的概率 p，画成两色区域 + 边界线
    xs = np.linspace(lim[0], lim[1], 400)
    ys = np.linspace(lim[0], lim[1], 400)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    p = proba(model, grid).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=[0, 0.5, 1], colors=["#ffd2d2", "#d2d2ff"], alpha=0.55)
    ax.contour(xx, yy, p, levels=[0.5], colors="black", linewidths=1.6)

    # 训练点（实心圆）
    ax.scatter(X_tr[y_tr.squeeze() == 0, 0], X_tr[y_tr.squeeze() == 0, 1],
               c="red", s=30, marker="o", edgecolors="k", linewidths=0.6, zorder=3)
    ax.scatter(X_tr[y_tr.squeeze() == 1, 0], X_tr[y_tr.squeeze() == 1, 1],
               c="blue", s=30, marker="o", edgecolors="k", linewidths=0.6, zorder=3)

    # 验证点（空心圆）
    ax.scatter(X_va[y_va.squeeze() == 0, 0], X_va[y_va.squeeze() == 0, 1],
               facecolors="none", edgecolors="red", s=40, marker="o", linewidths=0.9, zorder=3)
    ax.scatter(X_va[y_va.squeeze() == 1, 0], X_va[y_va.squeeze() == 1, 1],
               facecolors="none", edgecolors="blue", s=40, marker="o", linewidths=0.9, zorder=3)

    # 验证集里分错的点，打黑色大 X
    p_va = proba(model, X_va)
    wrong = ((p_va >= 0.5).float() != y_va.squeeze())
    if wrong.any():
        ax.scatter(X_va[wrong, 0], X_va[wrong, 1], s=200, marker="x",
                   c="black", linewidths=2.4, zorder=4)

    a = acc(model, X_tr, y_tr)
    b = acc(model, X_va, y_va)
    ax.set_title(f"{title}\ntrain {a:.3f}   val {b:.3f}   gap {a - b:+.3f}", fontsize=11)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))

    configs = [
        ("overlap (noise=1.0)", dict(noise=1.0, c0=(-1, -1), c1=(1, 1))),
        ("separable (noise=0.5)", dict(noise=0.5, c0=(-2, -2), c1=(2, 2))),
    ]

    for row, (name, cfg) in enumerate(configs):
        X, y = make_data(400, seed=0, **cfg)
        X_tr, y_tr, X_va, y_va = split(X, y)

        small = train_mlp(X_tr, y_tr, hidden=4, epochs=8000)
        big = train_mlp(X_tr, y_tr, hidden=128, epochs=8000)

        draw(axes[row, 0], small, X_tr, y_tr, X_va, y_va, f"{name}  |  small (hidden=4)")
        draw(axes[row, 1], big, X_tr, y_tr, X_va, y_va, f"{name}  |  big (hidden=128)")

    fig.suptitle("Decision boundary: small vs big model, overlap vs separable data",
                 fontsize=13, y=0.99)
    fig.text(0.5, 0.02,
             "red=class0  blue=class1   solid dot=train   hollow circle=val   black X=val wrong",
             ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig("Day10_过拟合可视化.png", dpi=110)
    print("saved: Day10_过拟合可视化.png")
