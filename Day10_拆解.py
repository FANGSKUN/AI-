# -*- coding: utf-8 -*-
"""
Day10 拆解：一个神经元 = 一道折痕，折痕相加 = 复杂折线
======================================================
左列：1 个神经元  out = ReLU(x+y) - 1   （一道折痕，边界是直线）
右列：2 个神经元  out = ReLU(x) + ReLU(y) - 1 （两道折痕，边界是 V 形）

上图 = 3D 曲面（模型输出），下图 = 2D 俯视（决策边界）
绿色虚线 = 折痕（ReLU 的拐点线），黑色实线 = 决策边界（输出=0）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

xs = np.linspace(-2, 2, 300)
ys = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(xs, ys)

Z1 = np.maximum(0, X + Y) - 1.0            # 1 个神经元：一道折痕
Z2 = np.maximum(0, X) + np.maximum(0, Y) - 1.0  # 2 个神经元：两道折痕

fig = plt.figure(figsize=(12, 10))

# ---------- 左上：1 个神经元 3D ----------
ax = fig.add_subplot(2, 2, 1, projection="3d")
ax.plot_surface(X, Y, Z1, cmap="coolwarm", alpha=0.85, linewidth=0)
# 折痕 x+y=0（在 z=-1 处沿折痕走）
ax.plot(xs, -xs, -np.ones_like(xs), color="green", lw=2.5, linestyle="--", label="hinge x+y=0")
# 边界 x+y=1（z=0）
ax.plot(xs, 1 - xs, np.zeros_like(xs), color="black", lw=3.5, label="boundary (out=0)")
ax.set_zlim(-2, 2)
ax.set_title("1 neuron:  out = ReLU(x+y) - 1\none hinge -> straight boundary", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("out")
ax.legend(fontsize=8)
ax.view_init(elev=28, azim=-60)

# ---------- 右上：2 个神经元 3D ----------
ax = fig.add_subplot(2, 2, 2, projection="3d")
ax.plot_surface(X, Y, Z2, cmap="coolwarm", alpha=0.85, linewidth=0)
# 折痕 x=0 和 y=0
ax.plot(np.zeros_like(ys), ys, -np.ones_like(ys), color="green", lw=2, linestyle="--")
ax.plot(xs, np.zeros_like(xs), -np.ones_like(xs), color="green", lw=2, linestyle="--", label="hinges x=0, y=0")
# 边界 V：射线 x=1(y<=0) + 射线 y=1(x<=0) + 斜边 x+y=1(第一象限)
yy_neg = np.linspace(-2, 0, 50)
xx_neg = np.linspace(-2, 0, 50)
tt = np.linspace(0, 1, 50)
ax.plot(np.ones_like(yy_neg), yy_neg, np.zeros_like(yy_neg), color="black", lw=3.5, label="boundary (V)")
ax.plot(xx_neg, np.ones_like(xx_neg), np.zeros_like(xx_neg), color="black", lw=3.5)
ax.plot(tt, 1 - tt, np.zeros_like(tt), color="black", lw=3.5)
ax.set_zlim(-2, 2)
ax.set_title("2 neurons:  out = ReLU(x) + ReLU(y) - 1\ntwo hinges -> V-shaped boundary", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("out")
ax.legend(fontsize=8)
ax.view_init(elev=28, azim=-60)

# ---------- 左下：1 个神经元 2D 俯视 ----------
ax = fig.add_subplot(2, 2, 3)
ax.contourf(X, Y, Z1, levels=[-10, 0, 10], colors=["#ffd2d2", "#d2d2ff"], alpha=0.6)
ax.plot(xs, -xs, color="green", lw=2, linestyle="--", label="hinge x+y=0")
ax.plot(xs, 1 - xs, color="black", lw=3, label="boundary x+y=1")
ax.set_title("1 neuron (top view): straight line", fontsize=10)
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
ax.legend(fontsize=8)
ax.set_xlabel("x"); ax.set_ylabel("y")

# ---------- 右下：2 个神经元 2D 俯视 ----------
ax = fig.add_subplot(2, 2, 4)
ax.contourf(X, Y, Z2, levels=[-10, 0, 10], colors=["#ffd2d2", "#d2d2ff"], alpha=0.6)
ax.plot(np.zeros_like(ys), ys, color="green", lw=2, linestyle="--", label="hinges x=0, y=0")
ax.plot(xs, np.zeros_like(xs), color="green", lw=2, linestyle="--")
ax.plot(np.ones_like(yy_neg), yy_neg, color="black", lw=3, label="boundary (V)")
ax.plot(xx_neg, np.ones_like(xx_neg), color="black", lw=3)
ax.plot(tt, 1 - tt, color="black", lw=3)
ax.set_title("2 neurons (top view): V shape", fontsize=10)
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
ax.legend(fontsize=8)
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("Day10_拆解.png", dpi=110)
print("saved: Day10_拆解.png")
