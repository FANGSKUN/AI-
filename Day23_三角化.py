# -*- coding: utf-8 -*-
r"""
Day23 项目：三角化——从 2D 反推 3D（Phase 4 继续，SLAM 的灵魂）
================================================================
Day22 学的是「正向」：3D 世界点 → 2D 像素（投影）。
今天反着走：已知 2D 像素，反推出它对应的 3D 点。

核心难点：一个像素 (u,v) 背后是【一整条光线】上的无数个 3D 点（深度未知），
单只眼测不出距离。解法：用【两个视角】看同一个点，两条光线相交，交点就是 3D 点
——这就是三角化（triangulation），SLAM 从图像恢复 3D 结构的基础。

关键副产品：双目视差测距公式
    深度 z = fx · 基线b / 视差(u1-u2)

今天任务：
    1. 反投影：像素 (u,v) → 世界坐标系里的一条光线（光心 C + 方向 d）；
    2. 三角化：两条光线求最近点，反解 3D 坐标；
    3. 视差测深：用 z = fx·b/(u1-u2) 验证深度。

运行：
    conda activate ai
    python Day23_自检.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401（注册 3D 投影）

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 1. 内参 + 投影（复用 Day22 的投影链） ============

def make_K(fx=300, fy=300, cx=320, cy=240):
    """3x3 内参矩阵（含齐次行 [0,0,1]）。"""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


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


# ============ 2. 反投影（像素 → 光线） ============

def backproject(u, v, K3, R, t):
    """像素 (u,v) → 世界坐标系里的一条光线，返回 (光心 C, 方向 d)。

    像素反推相机坐标方向：d_cam = [(u-cx)/fx, (v-cy)/fy, 1]（取深度 1 作参考）。
    """
    fx, fy = K3[0, 0], K3[1, 1]
    cx, cy = K3[0, 2], K3[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

    # TODO 1a：世界坐标方向（旋转的逆 = 转置）
    d_world = d_cam.T
    d_world = d_world / np.linalg.norm(d_world)   # 单位化，s 就是真实距离

    # TODO 1b：光心世界坐标（由 t = -R@C 反推 C = -R.T @ t）
    C = -R.T @ t
    return C, d_world


# ============ 3. 三角化（两条光线求最近点） ============

def triangulate(C1, d1, C2, d2):
    """两条光线 (C1,d1)、(C2,d2) 求最近点，返回 3D 点。

    光线 1: C1 + s1·d1；光线 2: C2 + s2·d2。
    无噪声时两光线精确相交；有噪声时找「最接近两条光线」的点（公垂线中点）。
    """
    w0 = C1 - C2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    dd = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b

    # TODO 2a：光线 1 的参数 s1（最小二乘解）
    s1 = (b*e - c*dd)/denom
    # TODO 2b：光线 2 的参数 s2
    s2 = (a*e - b*dd)/denom

    X1 = C1 + s1 * d1
    X2 = C2 + s2 * d2
    return (X1 + X2) / 2.0          # 取两最近点的中点


# ============ 4. 视差测深 ============

def disparity_depth(u1, u2, fx, b):
    """双目视差测深度：左像素 u1、右像素 u2、焦距 fx、基线 b。"""
    # TODO 3：深度 z = fx·b / 视差；视差 = u1 - u2（左减右）
    z = fx*b/(u1-u2)
    return z


# ============ 5. 可视化 + 主流程 ============

def main():
    K3 = make_K(fx=300, fy=300, cx=320, cy=240)
    b = 0.5                                # 基线：两相机 x 方向距离

    # 两个相机：左相机在原点朝 +z；右相机在 x=b 处朝 +z
    R1 = np.eye(3); t1 = np.zeros(3)               # 左相机，光心 C1=(0,0,0)
    R2 = np.eye(3); t2 = np.array([-b, 0.0, 0.0])  # 右相机，光心 C2=(b,0,0)

    # 造一个已知 3D 点
    X_true = np.array([[0.3, -0.2, 3.0]])

    # ① 投影到两个相机
    uv1 = project(X_true, K3, R1, t1)[0]           # (u1, v1)
    uv2 = project(X_true, K3, R2, t2)[0]           # (u2, v2)

    # ② 反投影两条光线
    C1, d1 = backproject(uv1[0], uv1[1], K3, R1, t1)
    C2, d2 = backproject(uv2[0], uv2[1], K3, R2, t2)

    # ③ 三角化还原
    X_est = triangulate(C1, d1, C2, d2)

    # ④ 视差测深
    z_disp = disparity_depth(uv1[0], uv2[0], K3[0, 0], b)

    print("=" * 56)
    print(f"原始 3D 点:   {X_true[0]}")
    print(f"三角化还原:   {X_est}")
    print(f"还原误差:     {np.linalg.norm(X_est - X_true[0]):.2e}")
    print(f"视差测深 z:   {z_disp:.4f}（真实深度 {X_true[0, 2]}）")
    print("=" * 56)

    # 可视化：光心 + 两条光线 + 交点
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    s_max = 4.0
    for C, d, color in [(C1, d1, "b"), (C2, d2, "g")]:
        line = np.array([C + s * d for s in np.linspace(0, s_max, 60)])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, lw=2)
    ax.scatter([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]],
               c="r", s=80, marker="o", label="光心")
    ax.scatter([X_est[0]], [X_est[1]], [X_est[2]],
               c="k", s=120, marker="*", label="三角化点")
    ax.scatter([X_true[0, 0]], [X_true[0, 1]], [X_true[0, 2]],
               c="y", s=50, marker="o", label="原始点")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("三角化：两条光线相交，还原 3D 点")
    ax.legend()
    plt.tight_layout()
    plt.savefig("可视化图/Day23_三角化.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("已保存：可视化图/Day23_三角化.png")


if __name__ == "__main__":
    main()
