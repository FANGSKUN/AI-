# -*- coding: utf-8 -*-
"""Day24 自检：验证 3 个 TODO（反对称矩阵 / 本质矩阵 / 归一化坐标 + 对极约束）。

运行：
    conda activate ai
    python Day24_自检.py
"""
import numpy as np
import Day24_对极几何 as mod


def c1_反对称():
    t = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    prod = mod.skew(t) @ v
    expected = np.cross(t, v)
    assert np.allclose(prod, expected, atol=1e-12), \
        f"skew(t)@v 应等于 t×v（{expected}），实际 {prod}。检查 TODO1。"
    return f"TODO1 ✓ skew(t)@v == t×v = {expected.tolist()}"


def c2_本质矩阵():
    R = mod.rot_y(0.3)
    t = np.array([-0.5, 0.0, 0.0])
    E = mod.essential(R, t)
    assert E.shape == (3, 3), f"E 应为 (3,3)，实际 {E.shape}。检查 TODO2。"
    assert np.allclose(E, mod.skew(t) @ R), "E 应等于 [t]× @ R。检查 TODO2。"
    return "TODO2 ✓ E = [t]× @ R，形状 (3,3)"


def c3_对极约束():
    K3 = mod.make_K()
    R1 = np.eye(3); t1 = np.zeros(3)
    R2 = mod.rot_y(np.radians(10)); t2 = np.array([-0.5, 0.0, 0.0])
    X = np.array([[0.3, -0.2, 3.0]])
    x1 = mod.project(X, K3, R1, t1)[0]
    x2 = mod.project(X, K3, R2, t2)[0]
    xh1 = mod.normalize(x1, K3)
    xh2 = mod.normalize(x2, K3)
    E = mod.essential(R2, t2)
    val = xh2 @ E @ xh1
    assert abs(val) < 1e-6, f"对极约束应≈0，实际 {val:.2e}。检查 TODO2/TODO3。"
    return f"TODO3 ✓ 对极约束 x̂2ᵀ E x̂1 = {val:.2e} ≈ 0"


def main():
    checks = [c1_反对称, c2_本质矩阵, c3_对极约束]
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
