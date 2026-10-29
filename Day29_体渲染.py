# Day 29：体渲染（Volume Rendering）—— NeRF 的第二大创新
# 目标：理解「沿光线积分」——从密度场 + 颜色场，算出光线最终的颜色
# 这 20 行不到的公式，就是 NeRF 能「从 3D 神经场拍出照片」的核心

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 第 1 步：定义一条光线上的「密度场」和「颜色场」=================
def make_fields():
    """沿一条光线采样 N 个点，给每个点一个密度 sigma 和颜色 color。
    场景：两个「物体」一前一后——前面的（近）是红色，后面的（远）是蓝色。"""
    N = 128
    x = np.linspace(0.0, 1.0, N)          # 光线上的位置：0 = 相机，1 = 远处
    delta = x[1] - x[0]                   # 采样间距 δ

    # 密度：两个高斯峰 = 两个物体（sigma 越大越不透明）
    sigma = (
        10.0 * np.exp(-((x - 0.35) ** 2) / 0.01) +   # 前面的红色物体（高密度）
        6.0  * np.exp(-((x - 0.70) ** 2) / 0.01)     # 后面的蓝色物体
    )

    # 颜色：每个点一个 RGB（前面的物体偏红，后面偏蓝，物体之外接近黑）
    color = np.zeros((N, 3))
    color[:, 0] = 1.0 * np.exp(-((x - 0.35) ** 2) / 0.01)   # R 通道
    color[:, 2] = 1.0 * np.exp(-((x - 0.70) ** 2) / 0.01)   # B 通道

    return x, delta, sigma, color

# ================= 第 2 步：体渲染（沿光线积分）=================
def volume_render(sigma, color, delta):
    """核心公式：C = Σ_i T_i · α_i · c_i
    提示：sigma (N,)，color (N,3)，delta 是标量。
    TODO 1：alpha = 每段的不透明度  1 - exp(-sigma * delta)
    TODO 2：T = 透射率，光走到「第 i 段之前」还剩多少
            做法：先算 exp(-cumsum(sigma * delta))，这是「包含第 i 段」的累积，
            要「不包含第 i 段」，所以往右移一位，把第一个 T 设成 1。
            提示：np.concatenate([[1.0], 某个[:-1]])
    TODO 3：C = 加权叠加。注意 color 是 (N,3)，要按「颜色那一维」求和 sum(axis=0)，
            并且 alpha、T 要广播成 (N,1)（用 [:, None]）。
    返回：C (3,), alpha (N,), T (N,)
    """
    # ---- 在这里填 ----
    alpha = 1-np.exp(-sigma * delta)
    T_含自己 = np.exp(-np.cumsum(sigma * delta)) #这里右移一位T是因为alpha对应的应该是上一个T（没有穿过时）
    T = np.concatenate([[1.0], T_含自己[:-1]])
    C = (T[:, None] * alpha[:, None] * color).sum(axis=0)
    # -------------------
    return C, alpha, T

# ================= 第 3 步：可视化 =================
def main():
    x, delta, sigma, color = make_fields()
    C, alpha, T = volume_render(sigma, color, delta)

    print(f"体渲染出的最终颜色 RGB = ({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    # 左：密度场 sigma(x)
    axes[0].plot(x, sigma, color="black", lw=2)
    axes[0].set_title("密度场 σ(x)：前面红物体 + 后面蓝物体")
    axes[0].set_xlabel("光线位置（0 = 相机）")
    axes[0].set_ylabel("密度 σ")
    # 中：透射率 T(x)
    axes[1].plot(x, T, color="purple", lw=2)
    axes[1].set_title("透射率 T(x)：光走到这里还剩多少")
    axes[1].set_xlabel("光线位置")
    axes[1].set_ylim(0, 1.1)
    # 右：最终颜色
    axes[2].add_patch(plt.Rectangle((0, 0), 1, 1, color=C))
    axes[2].set_title(f"最终颜色 RGB=({C[0]:.2f},{C[1]:.2f},{C[2]:.2f})")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.tight_layout()
    plt.savefig("可视化图/Day29_体渲染.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()
