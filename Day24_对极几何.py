# -*- coding: utf-8 -*-
r"""
Day24 项目：对极几何——本质矩阵 E + 对极约束（Phase 4 继续，SLAM 定位地基）
============================================================================
Day23 三角化假设「相机位姿已知」。今天反过来：相机动了但不知道它怎么动的，
怎么从两张图的几何关系入手？——这就是对极几何（epipolar geometry）。

核心事实：一个 3D 点 X 和两个光心 C1、C2 三点共面（极平面）。
极平面与两个图像平面相交，得到两条「极线」l1、l2：
    - X 在相机1 的投影 x1 落在极线 l1 上
    - X 在相机2 的投影 x2 落在极线 l2 上

关键约束（对极约束）：给定 x1，x2 一定在极线 l2 上。编码这个几何的是
【本质矩阵 E】（标定相机）：
    E = [t]× @ R           （[t]× 是平移向量的反对称矩阵）
    x̂2ᵀ @ E @ x̂1 = 0        （x̂ 是归一化坐标 = 反投影的光线方向）

今天任务：
    1. 构造反对称矩阵 [t]×；
    2. 构造本质矩阵 E = [t]× @ R；
    3. 归一化坐标（像素 → x̂ = K⁻¹x）；
    4. 验证对极约束 x̂2ᵀ E x̂1 = 0，并画出极线。

运行：
    conda activate ai
    python Day24_自检.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 1. 基础工具（内参 / 旋转 / 投影） ============

def make_K(fx=300, fy=300, cx=320, cy=240):
    """3x3 内参矩阵（含齐次行 [0,0,1]）。"""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def rot_y(theta):
    """绕 y 轴旋转 theta 弧度的 3x3 旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def project(points_world, K3, R, t):
    """世界点 (N,3) → 图像坐标 (N,2)：P = K3 @ [R|t] + 齐次除法。"""
    N = points_world.shape[0]
    pts_homo = np.concatenate([points_world, np.ones((N, 1))], axis=1)
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K3 @ Rt
    p_homo = (P @ pts_homo.T).T
    s = p_homo[:, 2]
    u = p_homo[:, 0] / s
    v = p_homo[:, 1] / s
    return np.stack([u, v], axis=1)


# ============ 2. 对极几何核心 ============

def skew(t):
    """平移向量 t=(tx,ty,tz) 的 3x3 反对称矩阵 [t]×。

    性质：skew(t) @ v == np.cross(t, v)（矩阵乘法 == 叉乘）。
    """
    tx, ty, tz = t
    # TODO 1：填 3x3 反对称矩阵（对角线为 0，其余反对称）
    return np.array([
        [0,-tz,ty],
        [tz,0,-tx],
        [-ty,tx,0],
    ])


def essential(R, t):
    """本质矩阵 E = [t]× @ R（编码相机1 → 相机2 的相对运动）。"""
    # TODO 2：反对称矩阵 左乘 旋转矩阵
    E = skew(t)@R
    return E


def normalize(pixel, K3):
    """像素 (u,v) → 归一化坐标 x̂ = K⁻¹ @ [u,v,1]（即反投影的光线方向）。"""
    fx, fy = K3[0, 0], K3[1, 1]
    cx, cy = K3[0, 2], K3[1, 2]
    u, v = pixel
    # TODO 3：K⁻¹ @ [u,v,1] = [(u-cx)/fx, (v-cy)/fy, 1]
    xh = [(u-cx)/fx, (v-cy)/fy, 1]
    return xh


# ============ 3. 可视化 + 主流程 ============

def main():
    K3 = make_K(fx=300, fy=300, cx=320, cy=240)
    b = 0.5

    # 相机1 = 世界（在原点朝 +z）；相机2 在 x=b 处，绕 y 轴转 10°
    R1 = np.eye(3); t1 = np.zeros(3)
    R2 = rot_y(np.radians(10)); t2 = np.array([-b, 0.0, 0.0])

    # 一个已知 3D 点
    X = np.array([[0.3, -0.2, 3.0]])

    # ① 投影到两个相机
    x1 = project(X, K3, R1, t1)[0]          # 相机1 像素 (u1,v1)
    x2 = project(X, K3, R2, t2)[0]          # 相机2 像素 (u2,v2)

    # ② 归一化坐标
    xh1 = normalize(x1, K3)
    xh2 = normalize(x2, K3)

    # ③ 本质矩阵（相对位姿 = 相机2 外参，因为相机1=世界）
    E = essential(R2, t2)

    # ④ 验证对极约束（归一化坐标）：x̂2ᵀ E x̂1 应 ≈ 0
    val = xh2 @ E @ xh1
    print("=" * 56)
    print(f"对极约束 x̂2ᵀ E x̂1 = {val:.2e}（应 ≈ 0）")

    # ⑤ 极线（像素空间）：F = K⁻ᵀ E K⁻¹，l2 = F @ x1
    Kinv = np.linalg.inv(K3)
    F = Kinv.T @ E @ Kinv                      # 基础矩阵（像素空间）
    x1_h = np.array([x1[0], x1[1], 1.0])
    x2_h = np.array([x2[0], x2[1], 1.0])
    l2 = F @ x1_h                              # 像素空间的极线
    a, b_coef, c = l2
    resid = x2_h @ l2                          # x2 在极线上：应 ≈ 0
    print(f"极线 l2（像素空间）= ({a:.4f}, {b_coef:.4f}, {c:.4f})")
    print(f"x2 在极线上：{resid:.2e}（应 ≈ 0）")
    print("=" * 56)

    # 可视化：在相机2 的图像上画极线 + x2 点
    fig, ax = plt.subplots(figsize=(6, 5))
    uu = np.linspace(0, 640, 100)
    # 极线 a·u + b·v + c = 0 → v = -(a·u + c)/b
    vv = -(a * uu + c) / b_coef
    ax.plot(uu, vv, "r-", lw=2, label="极线 l2")
    ax.scatter([x2[0]], [x2[1]], c="b", s=60, zorder=5, label="x2")
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.set_title("对极约束：x2 落在极线 l2 上")
    ax.legend()
    plt.tight_layout()
    plt.savefig("可视化图/Day24_对极约束.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("已保存：可视化图/Day24_对极约束.png")


if __name__ == "__main__":
    main()
