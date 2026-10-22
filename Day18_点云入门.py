# -*- coding: utf-8 -*-
r"""
Day18 项目：3D 视觉开场——点云入门（Phase 4 起航）
=================================================
从今天起，机器人的「眼睛」要从 2D 照片升级到 3D 世界。

一张 2D 照片是 3D 世界拍扁的投影，丢掉了「深度」——机器人不知道
"这个杯子离我多远、在空间哪个位置"，就没法伸手去抓。

3D 数据的 4 种表示：
  1. 点云 point cloud：一堆 3D 坐标点 (N, 3)，LiDAR/深度相机直接产出
  2. 体素 voxel：3D 版"像素"，空间切成小方块
  3. 网格 mesh：顶点 + 三角面，游戏/动画常用
  4. 深度图 depth map：每个像素存"这个点离相机多远"

点云是最常用的（传感器直接给），今天先玩点云。

运行：
    conda activate ai
    python Day18_自检.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def make_sphere(n=1000, r=1.0):
    """生成一个「球面点云」：(N, 3) 矩阵，N 个点，每个点 (x, y, z)。

    思路：用「球坐标」随机撒点，再转成「直角坐标」。
    球坐标 (r, θ, φ)：
      θ (theta) = 方位角，范围 0 ~ 2π
      φ (phi)   = 极角，范围 0 ~ π
    直角坐标转换公式（物理里你熟）：
      x = r·sin(φ)·cos(θ)
      y = r·sin(φ)·sin(θ)
      z = r·cos(φ)
    """
    theta = np.random.uniform(0, 2 * np.pi, n)   # 方位角
    phi = np.random.uniform(0, np.pi, n)          # 极角
    # TODO 1：按上面的公式补全 x / y / z（填 theta 或 phi）
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.stack([x, y, z], axis=1)   # 拼成 (N, 3)
    return points


def centroid(points):
    """点云的质心 = 所有点的「平均坐标」。

    points 形状 (N, 3)：N 个点、每个点 3 个坐标。
    要对 N 个点求平均 → 沿「点的那个轴」求 mean，得到 (3,) 的质心。
    """
    # TODO 2：沿哪个轴求 mean？（提示：axis=0 是"跨 N 个点"）
    return points.mean(axis=0)


def translate(points, t):
    """平移点云：每个点加上平移向量 t（t 是 (3,) 或 (N,3) 都行）。"""
    return points + t


def draw(points, title="点云", filename="可视化图/Day18_点云.png"):
    """3D 可视化：用 matplotlib 的 3D 散点图。"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # TODO 3：画 3D 散点。points 三列分别是 x/y/z，填列下标 0/1/2
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])   # 三个轴等比例，否则球会被拉扁
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    return filename


def main():
    points = make_sphere(1000)
    print(f"点云形状: {points.shape}")          # 应为 (1000, 3)
    print(f"质心: {centroid(points)}")          # 应接近 (0, 0, 0)
    draw(points, "球面点云")

    # 平移一下，对比看：质心跟着动
    moved = translate(points, np.array([2.0, 0.0, 0.0]))
    print(f"平移后质心: {centroid(moved)}")      # 应接近 (2, 0, 0)
    draw(moved, "平移后的球面点云", "可视化图/Day18_点云平移.png")


if __name__ == "__main__":
    main()
