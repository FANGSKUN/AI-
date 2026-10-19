# -*- coding: utf-8 -*-
"""
Day12 补充可视化：直观看「三划分 / k 折 / 选择偏差」
===================================================
3 张图：
  1. 三划分：400 个点按 train(蓝) / val(黄) / test(红) 上色，看随机交错
  2. k 折：5 张子图，轮流把 1 折标红当验证、其余灰当训练
  3. 选择偏差：每个 hidden 的 val vs test 柱状图，看「val 最优」≠「test 最优」
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Day12_评估方法 import make_data, split_three, train_model, accuracy

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

X, y = make_data(400, seed=0)
Xn = X.numpy()
yn = y.numpy().flatten()


# ============ 图 1：三划分 ============
def fig1():
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_three(X, y)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c="#3b82f6", s=16, label="train (60%)", alpha=0.85)
    ax.scatter(X_va[:, 0], X_va[:, 1], c="#f59e0b", s=30, label="val (20%)", edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(X_te[:, 0], X_te[:, 1], c="#ef4444", s=30, label="test (20%)", edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_title("train / val / test 三划分（随机交错，不按类别切）", fontsize=12)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("Day12_三划分.png", dpi=110)
    plt.close(fig)
    print("saved: Day12_三划分.png")


# ============ 图 2：k 折轮转 ============
def fig2():
    torch.manual_seed(7)
    n = X.shape[0]
    idx = torch.randperm(n)
    k = 5
    fold = n // k

    fig, axes = plt.subplots(1, k, figsize=(17, 3.6))
    for i in range(k):
        ax = axes[i]
        va = idx[i * fold:(i + 1) * fold]
        ax.scatter(Xn[:, 0], Xn[:, 1], c="#e5e7eb", s=9, edgecolors="none")          # 训练（灰）
        ax.scatter(Xn[va, 0], Xn[va, 1], c="#ef4444", s=13, edgecolors="none")       # 验证（红）
        ax.set_title(f"第 {i} 折当验证", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("k 折交叉验证：轮流拿 1 折（红）当验证，其余（灰）训练", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("Day12_k折.png", dpi=110)
    plt.close(fig)
    print("saved: Day12_k折.png")


# ============ 图 3：选择偏差 ============
def fig3():
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_three(X, y)
    hiddens = [4, 8, 16, 32, 64, 128]
    vals, tes = [], []
    for h in hiddens:
        m = train_model(X_tr, y_tr, hidden=h)
        vals.append(accuracy(m, X_va, y_va))
        tes.append(accuracy(m, X_te, y_te))

    x = np.arange(len(hiddens))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, vals, w, label="val（拿来选模型）", color="#93c5fd", edgecolor="#2563eb")
    b2 = ax.bar(x + w / 2, tes, w, label="test（最终成绩）", color="#fca5a5", edgecolor="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels([f"hidden={h}" for h in hiddens])
    ax.set_ylabel("准确率")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("选择偏差：val 挑出的「最优」在 test 上未必最优", fontsize=13)

    # 标注两个"最优"
    i_val = int(np.argmax(vals))
    i_te = int(np.argmax(tes))
    ax.annotate("val 最优", (x[i_val] - w / 2, vals[i_val]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=9, color="#1d4ed8")
    ax.annotate("test 最优", (x[i_te] + w / 2, tes[i_te]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=9, color="#b91c1c")
    if i_val != i_te:
        ax.annotate("不一致！", (x[i_val], 0.985), ha="center", fontsize=12,
                    color="#b91c1c", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("Day12_选择偏差.png", dpi=110)
    plt.close(fig)
    print("saved: Day12_选择偏差.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
