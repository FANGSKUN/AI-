# -*- coding: utf-8 -*-
r"""
Day10 项目：过拟合与泛化——train/val 划分 + 亲眼看到"训练好、验证差"
====================================================================
目标：
    1. 学会把数据切成训练集 / 验证集（train / val）
    2. 用小模型和大模型各训练一次，观察：
        大模型：训练准确率高（≈1.0），但验证集掉下来 → 过拟合
        小模型：训练/验证差不多 → 泛化好

把标了 TODO 的地方补全，然后运行 Day10_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day10_自检.py
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============ 1. 生成两团"有重叠"的数据 ============
# 注意：两团点故意让噪声大一点，这样即使最好也只能到 ~90%，
# 留出"过拟合"的空间（如果数据完全可分，就没法演示过拟合了）。

def make_data(n=400, seed=0, noise=1.0):
    torch.manual_seed(seed)
    n0 = n // 2
    X0 = torch.randn(n0, 2) * noise + torch.tensor([-1.0, -1.0])
    X1 = torch.randn(n - n0, 2) * noise + torch.tensor([1.0, 1.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n0, 1), torch.ones(n - n0, 1)], dim=0)
    return X, y


# ============ 2. 划分 train / val（TODO）============

def split(X, y, train_ratio=0.25, seed=7):
    """随机打乱后切成训练集和验证集，返回 (X_tr, y_tr, X_va, y_va)。"""
    # TODO 1：用 torch.randperm 打乱下标，按 train_ratio 切分
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n)          # 0..n-1 的随机排列
    k = int(n * train_ratio)         # 训练集大小
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


# ============ 3. 预测与评估（复习）============

def predict(model, X):
    with torch.no_grad():
        p = torch.sigmoid(model(X))
        return (p >= 0.5).float()


def accuracy(model, X, y):
    pred = predict(model, X)
    return (pred == y).float().mean().item()


# ============ 4. 训练 + 记录 train/val 准确率曲线 ============

def train_mlp(X_tr, y_tr, X_va, y_va, hidden=64, epochs=5000, lr=0.01, report_every=100):
    """训练 MLP，每隔 report_every 轮记录一次训练集和验证集准确率。

    返回 (model, train_accs, val_accs)。"""
    model = nn.Sequential(
        nn.Linear(2, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(torch.sigmoid(model(X_tr)), y_tr)
        loss.backward()
        optimizer.step()

        # TODO 2：每隔 report_every 轮记录一次准确率
        if epoch % report_every == 0:
           train_accs.append(accuracy(model, X_tr, y_tr))
           val_accs.append(accuracy(model, X_va, y_va))

    return model, train_accs, val_accs


# ============ 5. 主程序：小模型 vs 大模型 ============

def main():
    X, y = make_data()
    X_tr, y_tr, X_va, y_va = split(X, y)

    # 小模型（hidden=4）和大模型（hidden=64）
    small, s_tr, s_va = train_mlp(X_tr, y_tr, X_va, y_va, hidden=4)
    big, b_tr, b_va = train_mlp(X_tr, y_tr, X_va, y_va, hidden=64)

    print("=" * 50)
    print(f"训练集 {X_tr.shape[0]} 个样本，验证集 {X_va.shape[0]} 个样本")
    print(f"小模型(hidden=4) ：训练 {s_tr[-1]:.3f}  验证 {s_va[-1]:.3f}  差距 {s_tr[-1]-s_va[-1]:+.3f}")
    print(f"大模型(hidden=64)：训练 {b_tr[-1]:.3f}  验证 {b_va[-1]:.3f}  差距 {b_tr[-1]-b_va[-1]:+.3f}")
    print("差距越大 = 越过拟合（训练好、验证差）")
    print("=" * 50)

    xs = list(range(0, 5000, 100))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xs, s_tr, label="训练", color="blue")
    axes[0].plot(xs, s_va, label="验证", color="orange")
    axes[0].set_title("小模型 hidden=4（泛化好，两线贴近）")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("准确率"); axes[0].legend()

    axes[1].plot(xs, b_tr, label="训练", color="blue")
    axes[1].plot(xs, b_va, label="验证", color="orange")
    axes[1].set_title("大模型 hidden=64（过拟合，两线拉开）")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("准确率"); axes[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
