# -*- coding: utf-8 -*-
"""Day22 自检：验证 3 个 TODO（外参变换 / 齐次外参矩阵 / 完整投影链）。

运行：
    conda activate ai
    python Day22_自检.py
"""
import numpy as np
import Day22_外参与齐次坐标 as mod


def c1_外参变换():
    R = mod.rot_y(np.pi / 2)          # 绕 y 轴转 90°
    t = np.array([0.0, 0.0, 0.0])
    pts = np.array([[1.0, 0.0, 0.0]])  # 点 (1,0,0)
    cam = mod.world_to_cam(pts, R, t)
    # (1,0,0) 绕 y 转 90° → (0,0,-1)
    expected = np.array([[0.0, 0.0, -1.0]])
    assert np.allclose(cam, expected, atol=1e-6), \
        f"点(1,0,0)绕y转90°应为(0,0,-1)，实际 {cam}。检查 TODO1 的旋转。"
    return f"TODO1 ✓ 点(1,0,0)绕y转90° → {cam[0].tolist()}"


def c2_齐次外参():
    R = mod.rot_y(np.pi / 3)
    t = np.array([1.0, 2.0, 3.0])
    Rt = mod.make_extrinsic(R, t)
    assert Rt.shape == (3, 4), f"[R|t] 形状应为 (3,4)，实际 {Rt.shape}。检查 TODO2。"
    assert np.allclose(Rt[:3, :3], R), "前 3 列应为 R。检查 TODO2 的拼接。"
    assert np.allclose(Rt[:, 3], t), "第 4 列应为 t。检查 TODO2 的 t.reshape。"
    return "TODO2 ✓ [R|t] 形状 (3,4)，前3列=R、第4列=t"


def c3_完整投影链():
    K3 = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])
    R = mod.rot_y(np.radians(30))
    t = np.array([0.0, 0.0, 5.0])
    cube = mod.make_cube_vertices(1.0)

    # 一步：project_full（P = K3 @ [R|t]）
    p2d_full = mod.project_full(cube, K3, R, t)

    # 分步：world_to_cam + 内参投影
    cam = mod.world_to_cam(cube, R, t)
    fx, fy = K3[0, 0], K3[1, 1]
    cx, cy = K3[0, 2], K3[1, 2]
    x = fx * cam[:, 0] / cam[:, 2] + cx
    y = fy * cam[:, 1] / cam[:, 2] + cy
    p2d_manual = np.stack([x, y], axis=1)

    err = np.abs(p2d_full - p2d_manual).max()
    assert np.allclose(p2d_full, p2d_manual, atol=1e-4), \
        f"一步投影应与分步一致，最大误差 {err:.2e}。检查 TODO3 的 P 和齐次除法。"
    return f"TODO3 ✓ 一步投影(P=K·[R|t]) == 分步(旋转平移+投影)，误差 {err:.1e}"


def main():
    checks = [c1_外参变换, c2_齐次外参, c3_完整投影链]
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
