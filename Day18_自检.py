# -*- coding: utf-8 -*-
"""Day18 自检：验证 3 个 TODO 是否填对。

运行：
    conda activate ai
    python Day18_自检.py
"""
import os
import numpy as np
import Day18_点云入门 as mod


def c1_点云形状():
    points = mod.make_sphere(500)
    assert points.shape == (500, 3), \
        f"make_sphere 返回形状应为 (500, 3)，实际 {points.shape}。检查 TODO1 的 stack 拼接。"
    return f"TODO1 ✓ 点云形状 {points.shape}"


def c2_球面点云半径():
    # 球面上的每个点到原点的距离都应等于 r（这才是「球面」）
    points = mod.make_sphere(1000, r=2.0)
    dist = np.linalg.norm(points, axis=1)
    assert np.allclose(dist, 2.0, atol=1e-6), \
        "球面上每个点到原点的距离应都等于 r。检查 TODO1 的球坐标公式（sin/cos 放错会得到别的形状）。"
    return "TODO1 ✓ 所有点到原点距离 = r（确实是球面）"


def c3_质心():
    points = mod.make_sphere(2000)
    c = mod.centroid(points)
    assert c.shape == (3,), f"质心应为 (3,)，实际 {c.shape}。检查 TODO2 的 axis。"
    assert np.all(np.abs(c) < 0.15), \
        f"球心在原点的球面点云，质心应接近 (0,0,0)，实际 {np.round(c, 3)}。检查 TODO2 的 axis。"
    return f"TODO2 ✓ 质心 {np.round(c, 3)} ≈ (0,0,0)"


def c4_平移():
    points = mod.make_sphere(1000)
    moved = mod.translate(points, np.array([3.0, 0.0, 0.0]))
    c = mod.centroid(moved)
    assert np.allclose(c, [3.0, 0.0, 0.0], atol=0.15), \
        f"平移 (3,0,0) 后质心应接近 (3,0,0)，实际 {np.round(c, 3)}。检查 translate/centroid。"
    return f"TODO2 ✓ 平移 (3,0,0) 后质心 ≈ {np.round(c, 2)}"


def c5_可视化():
    points = mod.make_sphere(1000)
    fn = mod.draw(points, "自检点云", "可视化图/Day18_自检.png")
    assert os.path.exists(fn), "3D 可视化图未生成。检查 TODO3 的 scatter 三列下标。"
    return "TODO3 ✓ 3D 可视化图已生成"


def main():
    checks = [c1_点云形状, c2_球面点云半径, c3_质心, c4_平移, c5_可视化]
    print("=" * 56)
    all_ok = True
    for fn in checks:
        try:
            print("✓ " + fn())
        except Exception as e:
            all_ok = False
            print(f"✗ {fn.__name__} 失败：{e}")
    print("=" * 56)
    print("🎉 全部通过！" if all_ok else "有未通过的项，按提示改 TODO 后重跑。")


if __name__ == "__main__":
    main()
