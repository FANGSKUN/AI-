# -*- coding: utf-8 -*-
r"""
Day25 项目：八点法 + RANSAC——从匹配点反解相机运动（Phase 4 继续，SLAM 定位核心）
====================================================================================
Day24 推出了对极约束  x̂2ᵀ E x̂1 = 0，E = [t]× @ R 编码相机相对运动。
今天反过来用：给一堆匹配点，反解出 E，再分解出 R、t。

八点法（Eight-Point Algorithm）三步：
    ① 每对匹配点 (x̂1, x̂2) 给一个线性方程：展开 x̂2ᵀ E x̂1 = 0
       → 一行 [u2u1, u2v1, u2, v2u1, v2v1, v2, u1, v1, 1] · e = 0
    ② 8 对点堆成 A e = 0，SVD 求 A 的零空间 → E（强制秩 2）
    ③ SVD(E) 分解出 4 个候选 (R,t)，用三角化验证选「让点都在相机前方」的那个

RANSAC（随机采样一致性）：匹配点里有误匹配时，随机抽 8 对解 E、数内点、
迭代保留内点最多的 E，最后用内点精算。

运行：
    conda activate ai
    python Day25_自检.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 1. 基础工具（复用 Day22/24） ============

def make_K(fx=300, fy=300, cx=320, cy=240):
    """3x3 内参矩阵。"""
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
    """世界点 (N,3) → 图像坐标 (N,2)。"""
    N = points_world.shape[0]
    pts_homo = np.concatenate([points_world, np.ones((N, 1))], axis=1)
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K3 @ Rt
    p_homo = (P @ pts_homo.T).T
    s = p_homo[:, 2]
    return np.stack([p_homo[:, 0] / s, p_homo[:, 1] / s], axis=1)


def normalize(pixel, K3):
    """像素 (u,v) → 归一化坐标 x̂ = [(u-cx)/fx, (v-cy)/fy, 1]。"""
    fx, fy = K3[0, 0], K3[1, 1]
    cx, cy = K3[0, 2], K3[1, 2]
    u, v = pixel
    return np.array([(u - cx) / fx, (v - cy) / fy, 1.0])


# ============ 2. 八点法核心（TODO） ============

def build_A(x1_norm, x2_norm):
    """N 对归一化坐标 → N×9 矩阵 A，满足 A e = 0（e 是 E 的 9 个元素）。

    对每对点：x̂2ᵀ E x̂1 = 0 展开成
    u2u1·e11 + u2v1·e12 + u2·e13 + v2u1·e21 + v2v1·e22 + v2·e23 + u1·e31 + v1·e32 + e33 = 0
    """
    A = []
    for (u1, v1, _), (u2, v2, _) in zip(x1_norm, x2_norm):
        # TODO 1：填这一行（对极约束展开的 9 个系数）
        row = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
        A.append(row)
    return np.array(A)


def solve_E(x1_norm, x2_norm):
    """八点法：N 对匹配点 → 本质矩阵 E（3×3，秩 2）。"""
    A = build_A(x1_norm, x2_norm)
    # SVD 求 A 的零空间：最小奇异值对应的右奇异向量（Vt 最后一行）
    _, _, Vt = np.linalg.svd(A)
    # TODO 2a：零空间向量
    e = Vt[-1]
    E = e.reshape(3, 3)
    # 强制秩 2：本质矩阵理论秩为 2，把最小奇异值置 0
    U, S, Vt2 = np.linalg.svd(E)
    # TODO 2b：把最小奇异值置 0
    S[-1] = 0
    E = U @ np.diag(S) @ Vt2
    return E


def decompose_E(E):
    """SVD 分解 E → 4 个候选 (R, t)。"""
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # TODO 3a：两个旋转候选（用 W 和 Wᵀ）
    R1 = U@W@Vt
    R2 = U@W.T@Vt
    # 确保是合法旋转（det = +1）
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    # TODO 3b：平移候选（U 的第三列，方向确定、尺度不确定）
    t = U[:,2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


# ============ 3. 三角化验证（辅助，不是 TODO） ============

def triangulate_depth(x1_norm, x2_norm, R, t):
    """用候选 (R,t) 三角化一对点，返回两相机深度 (s1, s2)。

    关系 s2·x̂2 = R·(s1·x̂1) + t  →  最小二乘解出 s1、s2。
    正确候选应让 s1 > 0 且 s2 > 0（点在两个相机前方）。
    """
    A = np.column_stack([R @ x1_norm, -x2_norm])   # 3×2
    s, *_ = np.linalg.lstsq(A, t, rcond=None)
    return s[0], s[1]


# ============ 4. 主流程 ============

def main():
    K3 = make_K()
    # 相机 1 = 世界；相机 2 在 x=0.5，绕 y 轴转 10°
    R1 = np.eye(3); t1 = np.zeros(3)
    R2_true = rot_y(np.radians(10)); t2_true = np.array([-0.5, 0.0, 0.0])

    # 8 个 3D 点（z 都在相机前方）
    X = np.array([
        [0.3, -0.2, 3.0], [-0.2, 0.3, 3.5], [0.1, 0.1, 2.5], [-0.3, -0.3, 4.0],
        [0.4, 0.2, 3.2], [-0.1, 0.4, 2.8], [0.2, -0.4, 3.7], [-0.4, -0.1, 3.3],
    ])
    x1 = project(X, K3, R1, t1)                    # 相机 1 像素
    x2 = project(X, K3, R2_true, t2_true)          # 相机 2 像素
    xh1 = np.array([normalize(p, K3) for p in x1])
    xh2 = np.array([normalize(p, K3) for p in x2])

    print("=" * 60)
    # ① 八点法解 E
    E = solve_E(xh1, xh2)
    resid = [xh2[i] @ E @ xh1[i] for i in range(8)]
    print(f"八点法解出 E，秩 = {np.linalg.matrix_rank(E)}（应 = 2）")
    print(f"8 个点对极约束残差最大 = {max(abs(r) for r in resid):.2e}（应 ≈ 0）")

    # ② 分解 4 个候选，三角化验证选正确的
    cands = decompose_E(E)
    good = []
    for R, t in cands:
        s1, s2 = triangulate_depth(xh1[0], xh2[0], R, t)
        if s1 > 0 and s2 > 0:
            good.append((R, t))
    print(f"4 个候选中让点在前方的有 {len(good)} 个（应 = 1）")
    R_est, t_est = good[0]
    print(f"恢复的 R 是否≈真实 R2：{np.allclose(R_est, R2_true, atol=1e-6)}")
    cos = np.dot(t_est, t2_true) / (np.linalg.norm(t_est) * np.linalg.norm(t2_true))
    print(f"恢复的 t 方向与真实 t2 夹角余弦 = {cos:.4f}（应 ≈ ±1）")
    print("=" * 60)

    # 可视化：左=相机1 的 8 点；右=相机2 的 8 点 + 各自极线
    Kinv = np.linalg.inv(K3)
    F = Kinv.T @ E @ Kinv
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        ax.set_xlim(0, 640); ax.set_ylim(480, 0)
        ax.set_xlabel("u"); ax.set_ylabel("v")
    axes[0].scatter(x1[:, 0], x1[:, 1], c="b", s=30)
    axes[0].set_title("相机 1 的 8 个点")
    axes[1].scatter(x2[:, 0], x2[:, 1], c="r", s=30, zorder=5)
    uu = np.linspace(0, 640, 50)
    for i in range(8):
        l2 = F @ np.array([x1[i, 0], x1[i, 1], 1.0])
        a, b, c = l2
        vv = -(a * uu + c) / (b + 1e-9)
        axes[1].plot(uu, vv, "-", lw=1, alpha=0.5)
    axes[1].set_title("相机 2 的 8 个点 + 极线（点都在极线上）")
    plt.tight_layout()
    plt.savefig("可视化图/Day25_八点法.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("已保存：可视化图/Day25_八点法.png")


if __name__ == "__main__":
    main()
