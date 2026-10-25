# -*- coding: utf-8 -*-
r"""
Day22 项目：外参 + 齐次坐标（Phase 4 继续，SLAM 坐标变换地基）
================================================================
Day21 假设相机"端端正正"摆在原点、朝 +z 看。但真实相机是"摆"在世界某个位置、
朝向某个方向的。描述"相机摆在哪、朝哪看"的就是【外参】：旋转 R + 平移 t。

关键 trick【齐次坐标】：给 3D 点加一个"1"，变成 (X,Y,Z,1) 四维，这样：
    - 平移（本来是加法 X+t）也能用【矩阵乘法】表示
    - 旋转 + 平移 + 投影，全部串成【一次矩阵乘法】

完整投影链（SLAM 的核心公式）：
    s·[x;y;1] = K · [R|t] · [X;Y;Z;1]
     图像       内参   外参    世界(齐次)

今天任务：
    1. 实现外参变换（世界 → 相机）；
    2. 把 R、t 组合成齐次外参矩阵 [R|t]；
    3. 用 P = K·[R|t] 一步完成「世界 → 图像」，并让相机绕立方体转一圈看投影变化。

运行：
    conda activate ai
    python Day22_自检.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 1. 造立方体（复用 Day21 思路） ============

def make_cube_vertices(half=1.0):
    """立方体 8 个顶点 (8, 3)，用二进制位表示 x/y/z 正负。"""
    vs = []
    for i in range(8):
        x = half if (i >> 0) & 1 else -half
        y = half if (i >> 1) & 1 else -half
        z = half if (i >> 2) & 1 else -half
        vs.append([x, y, z])
    return np.array(vs)


def make_cube_edges():
    """立方体 12 条边（两顶点只在一个轴上不同 = 汉明距离 1）。"""
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                edges.append((i, j))
    return edges


# ============ 2. 旋转矩阵 ============

def rot_y(theta):
    """绕 y 轴旋转 theta 弧度的 3x3 旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


# ============ 3. 外参 + 齐次坐标 + 投影 ============

def world_to_cam(points_world, R, t):
    """外参变换：世界坐标 (N,3) → 相机坐标 (N,3)：X_cam = R @ X_world + t。

    R: (3,3) 旋转矩阵；t: (3,) 平移向量。
    """
    # TODO 1：旋转 + 平移。
    #   提示：points_world @ R.T 完成旋转（每行一个点，右乘转置），
    #         + t 完成平移（numpy 广播，每个点都加同一个 t）。
    cam = points_world@R.T + t
    return cam


def make_extrinsic(R, t):
    """把 R(3x3) 和 t(3,) 组合成 3x4 的外参矩阵 [R | t]。"""
    # TODO 2：沿列拼接 R 和 t。
    #   提示：np.hstack 沿列拼接；t 是 (3,)，要 reshape 成 (3,1) 列向量。
    Rt = np.concatenate([R,t.reshape((3, 1))],axis=1)
    return Rt


def project_full(points_world, K3, R, t):
    """完整投影链：世界 → 图像，用齐次坐标一次矩阵乘法 P = K3 @ [R|t]。

    K3: 3x3 内参（含齐次行 [0,0,1]）。返回 (N,2) 图像坐标。
    """
    N = points_world.shape[0]
    # 世界点 → 齐次 (N,4)：加一列 1
    pts_homo = np.concatenate([points_world, np.ones((N, 1))], axis=1)

    # TODO 3a：投影矩阵 P = K3 @ 外参[R|t]，形状 (3,3)@(3,4) = (3,4)
    Rt = make_extrinsic(R, t)
    P = K3@Rt

    # 投影：P @ 齐次点，得到 [s*x, s*y, s]，注意要除以深度 s
    p_homo = (P @ pts_homo.T).T          # (N, 3)
    # TODO 3b：齐次除法——除以深度 s，得到真正的 (x, y)
    s = p_homo[:, 2]
    x = p_homo[:, 0]/s
    y = p_homo[:, 1]/s
    return np.stack([x, y], axis=1)


# ============ 4. 可视化 ============

def main():
    K3 = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])   # 3x3 内参
    cube = make_cube_vertices(1.0)                             # 立方体在世界原点
    edges = make_cube_edges()

    # 相机绕 y 轴转 4 个角度，看立方体的 2D 投影怎么变（外参的作用）
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for k, (deg, ax) in enumerate(zip([0, 30, 60, 90], axes.ravel())):
        theta = np.radians(deg)
        R = rot_y(theta)              # 相机绕 y 轴转 theta（等价于世界转 theta）
        t = np.array([0, 0, 5])       # 立方体在相机前方 z=5
        p2d = project_full(cube, K3, R, t)
        for i, j in edges:
            ax.plot([p2d[i, 0], p2d[j, 0]], [p2d[i, 1], p2d[j, 1]], "b-", lw=2)
        ax.scatter(p2d[:, 0], p2d[:, 1], c="r", s=30, zorder=5)
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_aspect("equal")
        ax.set_title(f"相机绕 y 轴转 {deg}°")
    plt.tight_layout()
    plt.savefig("可视化图/Day22_外参旋转.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("已保存：可视化图/Day22_外参旋转.png（4 个角度对比）")


if __name__ == "__main__":
    main()
