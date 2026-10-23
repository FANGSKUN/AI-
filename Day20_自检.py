# -*- coding: utf-8 -*-
"""Day20 自检：验证 3 个 TODO + 训练收敛 + 分割的「等变性」。

运行：
    conda activate ai
    python Day20_自检.py
"""
import numpy as np
import torch
import Day20_PointNet分割 as mod


def c1_哑铃数据():
    pts, lab = mod.make_dumbbell(256)
    assert pts.shape == (256, 3), \
        f"哑铃点云形状应为 (256, 3)，实际 {pts.shape}。检查 TODO1 的返回。"
    assert lab.shape == (256,), \
        f"标签形状应为 (256,)，实际 {lab.shape}。"
    assert set(np.unique(lab)) == {0, 1}, \
        f"标签应只有 0/1 两类，实际 {np.unique(lab)}。"
    # 上面的球(label 0)的 z 均值应 > 0，下面的球(label 1)应 < 0
    z0 = pts[lab == 0][:, 2].mean()
    z1 = pts[lab == 1][:, 2].mean()
    assert z0 > 0 and z1 < 0, \
        f"label 0 应是上面的球(z均值>0)、label 1 应是下面的球(z均值<0)，实际 {z0:.2f}/{z1:.2f}。检查 TODO1 的平移方向。"
    return f"TODO1 ✓ 哑铃点云 {pts.shape}，两球心 z 均值 {z0:+.2f}/{z1:+.2f}"


def c2_前向形状():
    model = mod.PointNetSeg().to(mod.device)
    x = torch.randn(4, 100, 3).to(mod.device)
    out = model(x)
    assert out.shape == (4, 100, 2), \
        f"分割输出应为 (B, N, 2)（每个点一个分数），实际 {tuple(out.shape)}。检查 TODO2/TODO3。"
    return f"TODO2/3 ✓ 前向输出 {tuple(out.shape)}（每点一个标签）"


def c3_等变性():
    """分割是「等变」：打乱点的顺序，预测跟着点走（不像分类那样不变）。"""
    model = mod.PointNetSeg().to(mod.device)
    pts, _ = mod.make_dumbbell(128)
    x = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(mod.device)  # (1, N, 3)
    with torch.no_grad():
        pred = model(x).squeeze(0).argmax(dim=-1)      # (N,)

    perm = torch.randperm(128)
    with torch.no_grad():
        pred_perm = model(x[:, perm, :]).squeeze(0).argmax(dim=-1)

    assert torch.equal(pred_perm, pred[perm]), \
        "分割应等变：打乱输入后，预测应按同样顺序重排。检查 TODO3 的跳跃连接（每点信息要跟着点走）。"
    return "TODO3 ✓ 分割等变性成立（预测跟着点走，不是整块一个标签）"


def c4_训练收敛():
    model = mod.PointNetSeg().to(mod.device)
    X, y = mod.make_dataset(n_each=60, n_points=256)
    accs = mod.train(model, X, y, epochs=40)
    assert accs[-1] > 0.9, \
        f"逐点准确率应 > 0.9，实际 {accs[-1]:.3f}。检查 PointNetSeg 结构 / 训练（二次边界约 30 轮「顿悟」，40 轮稳过）。"
    return f"训练 ✓ 最终逐点准确率 {accs[-1]:.3f}（成功给每个点分类）"


def main():
    checks = [c1_哑铃数据, c2_前向形状, c3_等变性, c4_训练收敛]
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
