# -*- coding: utf-8 -*-
"""Day21 自检：验证 3 个 TODO（投影公式 / 透视近大远小 / 焦距）。

运行：
    conda activate ai
    python Day21_自检.py
"""
import numpy as np
import Day21_相机几何 as mod


def c1_投影公式():
    K = np.array([[100, 0, 0], [0, 100, 0]])          # fx=fy=100，主点 (0,0)
    pts = np.array([[3.0, 4.0, 5.0]])                 # 一个 3D 点
    p2d = mod.project(pts, K)
    expected = np.array([[60.0, 80.0]])               # x=100*3/5=60, y=100*4/5=80
    assert np.allclose(p2d, expected), \
        f"投影应为 {expected}，实际 {p2d}。检查 TODO1 的公式 x=fx*X/Z+cx、y=fy*Y/Z+cy。"
    return f"TODO1 ✓ 投影 (3,4,5) → {p2d[0].tolist()}（x=100·3/5=60，y=100·4/5=80）"


def c2_近大远小():
    K = np.array([[300, 0, 320], [0, 300, 240]])
    near = mod.make_cube_vertices(1.0) + np.array([0, 0, 5])   # 近
    far  = mod.make_cube_vertices(1.0) + np.array([0, 0, 10])  # 远
    w_near = mod.project_width(near, K)
    w_far  = mod.project_width(far, K)
    assert w_near > w_far, \
        f"近(z=5)投影宽 {w_near:.1f} 应 > 远(z=10)投影宽 {w_far:.1f}。检查 TODO2 的 z 值。"
    return f"TODO2 ✓ 近大远小（近 {w_near:.1f}px > 远 {w_far:.1f}px）"


def c3_焦距():
    K_wide = np.array([[200, 0, 320], [0, 200, 240]])   # 广角（短焦距）
    K_tele = np.array([[400, 0, 320], [0, 400, 240]])   # 长焦（长焦距）
    cube = mod.make_cube_vertices(1.0) + np.array([0, 0, 5])
    w_wide = mod.project_width(cube, K_wide)
    w_tele = mod.project_width(cube, K_tele)
    assert w_tele > w_wide, \
        f"长焦(fx=400)投影宽 {w_tele:.1f} 应 > 广角(fx=200)投影宽 {w_wide:.1f}。检查 TODO3 的焦距值。"
    return f"TODO3 ✓ 焦距越大越放大（长焦 {w_tele:.1f}px > 广角 {w_wide:.1f}px）"


def main():
    checks = [c1_投影公式, c2_近大远小, c3_焦距]
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
