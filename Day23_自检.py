# -*- coding: utf-8 -*-
"""Day23 自检：验证 3 个 TODO（反投影 / 三角化 / 视差测深）。

运行：
    conda activate ai
    python Day23_自检.py
"""
import numpy as np
import Day23_三角化 as mod


def c1_反投影():
    K3 = mod.make_K()
    R = np.eye(3); t = np.zeros(3)
    X = np.array([[0.3, -0.2, 3.0]])
    uv = mod.project(X, K3, R, t)[0]
    C, d = mod.backproject(uv[0], uv[1], K3, R, t)
    # 光线应经过原始点：C + s·d ≈ X
    s = np.dot(X[0] - C, d)
    X_reproj = C + s * d
    err = np.linalg.norm(X_reproj - X[0])
    assert np.allclose(X_reproj, X[0], atol=1e-6), \
        f"反投影光线应经过原始点，误差 {err:.2e}。检查 TODO1。"
    return f"TODO1 ✓ 反投影光线经过原始点，误差 {err:.1e}"


def c2_三角化():
    K3 = mod.make_K()
    b = 0.5
    R1 = np.eye(3); t1 = np.zeros(3)
    R2 = np.eye(3); t2 = np.array([-b, 0.0, 0.0])
    X = np.array([[0.3, -0.2, 3.0]])
    uv1 = mod.project(X, K3, R1, t1)[0]
    uv2 = mod.project(X, K3, R2, t2)[0]
    C1, d1 = mod.backproject(uv1[0], uv1[1], K3, R1, t1)
    C2, d2 = mod.backproject(uv2[0], uv2[1], K3, R2, t2)
    X_est = mod.triangulate(C1, d1, C2, d2)
    err = np.linalg.norm(X_est - X[0])
    assert err < 1e-6, f"三角化应精确还原 3D 点，误差 {err:.2e}。检查 TODO2。"
    return f"TODO2 ✓ 三角化还原 3D 点，误差 {err:.1e}"


def c3_视差测深():
    K3 = mod.make_K()
    b = 0.5
    fx = K3[0, 0]
    X = np.array([[0.3, -0.2, 3.0]])
    uv_left = mod.project(X, K3, np.eye(3), np.zeros(3))[0]
    uv_right = mod.project(X, K3, np.eye(3), np.array([-b, 0.0, 0.0]))[0]
    z = mod.disparity_depth(uv_left[0], uv_right[0], fx, b)
    assert abs(z - X[0, 2]) < 1e-6, \
        f"视差深度 {z} 应≈真实深度 {X[0,2]}。检查 TODO3。"
    return f"TODO3 ✓ 视差测深 z={z:.4f}（真实深度 {X[0,2]}）"


def main():
    checks = [c1_反投影, c2_三角化, c3_视差测深]
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
