# -*- coding: utf-8 -*-
r"""
Day21 项目：相机几何——针孔相机模型（Phase 4 继续，SLAM / 3D 视觉地基）
=====================================================================
机器人用相机"看"世界，本质是把 3D 世界点投影成 2D 图像像素。这个投影就是
「针孔相机模型」——光线穿过一个小孔，把外面的 3D 世界映到成像平面。

核心公式（透视投影）：
    x = fx * X/Z + cx
    y = fy * Y/Z + cy

    (X,Y,Z) = 3D 点；fx/fy = 焦距（内参）；cx/cy = 主点（内参）；(x,y) = 图像坐标。
    除以 Z 就是「透视除法」——造成"近大远小"。

今天任务：
    1. 实现投影函数，把一个 3D 立方体投影成 2D 线框（直观看到"3D → 2D"）；
    2. 验证透视"近大远小"（同样的立方体，放得越远投影越小）；
    3. 验证焦距的作用（焦距越长，投影越"放大"= 长焦镜头）。

运行：
    conda activate ai
    python Day21_自检.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 1. 造 3D 立方体 ============

def make_cube_vertices(half=1.0):
    """立方体 8 个顶点 (8, 3)。用 0~7 的二进制位表示 x/y/z 的正负。"""
    vs = []
    for i in range(8):
        x = half if (i >> 0) & 1 else -half
        y = half if (i >> 1) & 1 else -half
        z = half if (i >> 2) & 1 else -half
        vs.append([x, y, z])
    return np.array(vs)


def make_cube_edges():
    """立方体 12 条边，每条边 = 一对顶点索引 (i, j)（两顶点只在一个轴上不同）。"""
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:   # 汉明距离 1 = 只有一轴不同
                edges.append((i, j))
    return edges


# ============ 2. 针孔相机投影 ============

def project(points, K):
    """针孔相机投影：3D 点 (N,3) → 2D 图像点 (N,2)。

    K: 内参矩阵 [[fx, 0, cx], [0, fy, cy]]（fx/fy 焦距，cx/cy 主点）。
    核心：x = fx*X/Z + cx，y = fy*Y/Z + cy，其中「除以 Z」是透视除法。
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # TODO 1：透视投影公式。提示：x = fx * X / Z + cx（除以 Z 造成近大远小）。
    x = fx * X / Z +cx
    y = fy * Y / Z +cy
    return np.stack([x, y], axis=1)


def project_width(points, K):
    """投影后 x 坐标的范围（像素宽度），用来比较「近大远小」和「焦距」"""
    p2d = project(points, K)
    return p2d[:, 0].max() - p2d[:, 0].min()


# ============ 3. 可视化 ============

def draw_projection(p2d, edges, filename="可视化图/Day21_投影.png"):
    """画出 2D 投影的立方体线框（12 条边）。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, j in edges:
        ax.plot([p2d[i, 0], p2d[j, 0]], [p2d[i, 1], p2d[j, 1]], "b-", lw=2)
    ax.scatter(p2d[:, 0], p2d[:, 1], c="r", s=40, zorder=5)
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)          # 图像坐标：原点在左上，y 向下
    ax.set_aspect("equal")
    ax.set_title("立方体的 2D 投影（针孔相机）")
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    return filename


# ============ 4. 主程序 ============

def main():
    K = np.array([[300, 0, 320], [0, 300, 240]])   # 内参：fx=fy=300，主点 (320,240)（640x480 图像中心）

    # 把立方体放在相机正前方 z=5，投影并画图
    cube = make_cube_vertices(1.0) + np.array([0, 0, 5])
    p2d = project(cube, K)
    draw_projection(p2d, make_cube_edges())
    print("已保存 2D 投影图：可视化图/Day21_投影.png")

    # TODO 2：验证透视「近大远小」。
    #   把两个同样大小的立方体放在 z=5（近）和 z=10（远），比较投影宽度。
    #   提示：透视除法「除以 Z」，Z 越大投影越小 → 近的立方体投影更宽。
    cube_near = make_cube_vertices(1.0) + np.array([0, 0, 5])   # 近：z=5
    cube_far  = make_cube_vertices(1.0) + np.array([0, 0, 10])   # 远：z=10
    w_near = project_width(cube_near, K)
    w_far  = project_width(cube_far, K)
    print(f"近(z=5)投影宽 {w_near:.1f}px，远(z=10)投影宽 {w_far:.1f}px")
    print(f"近大远小成立？ {w_near > w_far}")

    # TODO 3：验证焦距的作用——焦距越长，投影越「放大」（长焦镜头）。
    #   填两个不同的焦距：短焦距=广角（拍得广但小），长焦距=长焦（拍得窄但大）。
    fx_wide = 200   # 短焦距（广角），比如 200
    fx_tele = 400   # 长焦距（长焦），比如 400
    K_wide = np.array([[fx_wide, 0, 320], [0, fx_wide, 240]])
    K_tele = np.array([[fx_tele, 0, 320], [0, fx_tele, 240]])
    w_wide = project_width(cube_near, K_wide)
    w_tele = project_width(cube_near, K_tele)
    print(f"广角(fx={fx_wide})投影宽 {w_wide:.1f}px，长焦(fx={fx_tele})投影宽 {w_tele:.1f}px")
    print(f"焦距越大越放大？ {w_tele > w_wide}")


if __name__ == "__main__":
    main()
