# -*- coding: utf-8 -*-
"""
Day10 补充可视化：隐藏神经元个数 = 边界能折的次数
==================================================
同一批"有重叠"的数据，用 hidden = 1, 2, 4, 8, 16, 32 各训练一次，
画出决策边界。看边界怎么随着神经元变多而"越折越绕"。
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


def train_mlp(X_tr, y_tr, hidden=4, epochs=8000, lr=0.01):
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


def draw(ax, model, X_tr, y_tr, X_va, y_va, title, lim=(-4.0, 4.0)):
    xs = np.linspace(lim[0], lim[1], 400)
    ys = np.linspace(lim[0], lim[1], 400)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    p = proba(model, grid).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=[0, 0.5, 1], colors=["#ffd2d2", "#d2d2ff"], alpha=0.55)
    ax.contour(xx, yy, p, levels=[0.5], colors="black", linewidths=1.6)

    ax.scatter(X_tr[y_tr.squeeze() == 0, 0], X_tr[y_tr.squeeze() == 0, 1],
               c="red", s=22, marker="o", edgecolors="k", linewidths=0.5, zorder=3)
    ax.scatter(X_tr[y_tr.squeeze() == 1, 0], X_tr[y_tr.squeeze() == 1, 1],
               c="blue", s=22, marker="o", edgecolors="k", linewidths=0.5, zorder=3)

    a = acc(model, X_tr, y_tr)
    b = acc(model, X_va, y_va)
    ax.set_title(f"hidden={title}\ntrain {a:.2f}  val {b:.2f}", fontsize=10)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    X, y = make_data(400, seed=0, noise=1.0)
    X_tr, y_tr, X_va, y_va = split(X, y)

    hiddens = [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    for ax, h in zip(axes.ravel(), hiddens):
        m = train_mlp(X_tr, y_tr, hidden=h, epochs=8000)
        draw(ax, m, X_tr, y_tr, X_va, y_va, h)

    fig.suptitle("More hidden neurons = more folds in the decision boundary",
                 fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Day10_隐藏层折线.png", dpi=110)
    print("saved: Day10_隐藏层折线.png")
