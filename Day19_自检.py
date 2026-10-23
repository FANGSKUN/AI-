# -*- coding: utf-8 -*-
"""Day19 自检：验证 3 个 TODO + 训练收敛。

运行：
    conda activate ai
    python Day19_自检.py
"""
import numpy as np
import torch
import Day19_PointNet入门 as mod


def c1_立方体点云():
    pts = mod.make_cube_surface(2000)
    assert pts.shape == (2000, 3), \
        f"立方体点云形状应为 (2000, 3)，实际 {pts.shape}。检查 TODO1 的返回。"
    # 立方体表面的点：每个点至少有一个坐标 = ±half（在某个面上）
    on_surface = np.any(np.isclose(np.abs(pts), 1.0, atol=1e-6), axis=1)
    assert on_surface.all(), \
        "立方体面点云应每个点至少一个坐标 = ±1。检查 TODO1 的 6 个面坐标。"
    return f"TODO1 ✓ 立方体面点云 {pts.shape}，点都在表面"


def c2_前向形状():
    model = mod.PointNet().to(mod.device)
    x = torch.randn(4, 100, 3).to(mod.device)
    out = model(x)
    assert out.shape == (4, 2), \
        f"输出应为 (4, 2)，实际 {tuple(out.shape)}。检查 TODO2 的 max 的 dim。"
    return f"TODO2 ✓ 前向输出 {tuple(out.shape)}"


def c3_无序性():
    model = mod.PointNet().to(mod.device)
    pts = mod.make_sphere(256)
    diff = mod.permutation_diff(model, pts)
    assert diff < 1e-5, \
        f"打乱点顺序后输出差应 ≈ 0，实际 {diff}。检查 TODO2 的对称函数 / TODO3 的 diff。"
    return f"TODO3 ✓ 打乱点顺序后输出差 = {diff:.2e} ≈ 0（顺序无关）"


def c4_训练收敛():
    model = mod.PointNet().to(mod.device)
    X, y = mod.make_dataset(n_each=80, n_points=256)
    accs = mod.train(model, X, y, epochs=15)
    assert accs[-1] > 0.85, \
        f"最终准确率应 > 0.85，实际 {accs[-1]:.3f}。检查 PointNet 结构 / 训练。"
    return f"训练 ✓ 最终准确率 {accs[-1]:.3f}（成功区分球 vs 立方体）"


def main():
    checks = [c1_立方体点云, c2_前向形状, c3_无序性, c4_训练收敛]
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
