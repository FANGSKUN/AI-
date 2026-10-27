# -*- coding: utf-8 -*-
"""Day25 自检：验证 3 个 TODO（build_A / solve_E / decompose_E）。

运行：
    conda activate ai
    python Day25_自检.py
"""
import numpy as np
import Day25_八点法 as mod

# 复用 main 的 8 个 3D 点
POINTS = np.array([
    [0.3, -0.2, 3.0], [-0.2, 0.3, 3.5], [0.1, 0.1, 2.5], [-0.3, -0.3, 4.0],
    [0.4, 0.2, 3.2], [-0.1, 0.4, 2.8], [0.2, -0.4, 3.7], [-0.4, -0.1, 3.3],
])


def _scene():
    """造 8 对匹配点（相机 1 = 世界，相机 2 有相对位姿）。"""
    K3 = mod.make_K()
    R1 = np.eye(3); t1 = np.zeros(3)
    R2 = mod.rot_y(np.radians(10)); t2 = np.array([-0.5, 0.0, 0.0])
    x1 = mod.project(POINTS, K3, R1, t1)
    x2 = mod.project(POINTS, K3, R2, t2)
    xh1 = np.array([mod.normalize(p, K3) for p in x1])
    xh2 = np.array([mod.normalize(p, K3) for p in x2])
    return K3, R2, t2, xh1, xh2


def c1_build_A():
    x1 = np.array([[0.1, -0.2, 1.0], [0.3, 0.4, 1.0]])
    x2 = np.array([[-0.5, 0.6, 1.0], [0.7, -0.8, 1.0]])
    A = mod.build_A(x1, x2)
    assert A.shape == (2, 9), f"A 应为 (2,9)，实际 {A.shape}"
    expected = [-0.5 * 0.1, -0.5 * (-0.2), -0.5, 0.6 * 0.1, 0.6 * (-0.2), 0.6, 0.1, -0.2, 1.0]
    assert np.allclose(A[0], expected), f"第一行应为 {expected}，实际 {A[0]}。检查 TODO1。"
    return "TODO1 ✓ A 行向量 = 对极约束展开的 9 个系数"


def c2_solve_E():
    _, _, _, xh1, xh2 = _scene()
    E = mod.solve_E(xh1, xh2)
    assert E.shape == (3, 3), f"E 应为 (3,3)，实际 {E.shape}"
    assert np.linalg.matrix_rank(E) == 2, f"E 应秩 2，实际秩 {np.linalg.matrix_rank(E)}。检查 TODO2b。"
    for i in range(8):
        val = xh2[i] @ E @ xh1[i]
        assert abs(val) < 1e-6, f"第 {i} 点对极约束 {val:.2e} 应≈0。检查 TODO2。"
    return "TODO2 ✓ 八点法解出秩 2 的 E，8 点对极约束全 ≈ 0"


def c3_decompose_E():
    _, R2, t2, xh1, xh2 = _scene()
    E = mod.solve_E(xh1, xh2)
    cands = mod.decompose_E(E)
    assert len(cands) == 4, f"应 4 个候选，实际 {len(cands)}。检查 TODO3。"
    good = []
    for R, t in cands:
        s1, s2 = mod.triangulate_depth(xh1[0], xh2[0], R, t)
        if s1 > 0 and s2 > 0:
            good.append((R, t))
    assert len(good) == 1, f"应 1 个候选让点在前方，实际 {len(good)}。检查 TODO3。"
    R_est, t_est = good[0]
    assert np.allclose(R_est, R2, atol=1e-6), f"恢复的 R 应≈真实 R2。检查 TODO3a。"
    cos = np.dot(t_est, t2) / (np.linalg.norm(t_est) * np.linalg.norm(t2))
    assert abs(cos) > 0.99, f"恢复的 t 方向应≈真实 t2 方向，余弦 {cos:.4f}。检查 TODO3b。"
    return f"TODO3 ✓ 4 候选里 1 个正确（R 还原、t 方向余弦 {cos:.4f}）"


def main():
    checks = [c1_build_A, c2_solve_E, c3_decompose_E]
    print("=" * 60)
    all_ok = True
    for fn in checks:
        try:
            print("✓ " + fn())
        except Exception as e:
            all_ok = False
            print(f"✗ {fn.__name__} 失败：{e}")
    print("=" * 60)
    print("🎉 全部通过！" if all_ok else "有未通过的项，按提示改 TODO 后重跑。")


if __name__ == "__main__":
    main()
