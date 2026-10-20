# -*- coding: utf-8 -*-
"""
Day15 自检：CIFAR-10 彩色图像分类（3 通道 CNN）
运行：
    conda activate ai
    python Day15_自检.py
"""
import torch
import torch.nn as nn

import Day15_CIFAR10分类 as mod


def c1_模型含三个卷积层():
    m = mod.build_cnn()
    convs = [layer for layer in m if isinstance(layer, nn.Conv2d)]
    assert len(convs) == 3, f"三层 CNN 应该有 3 个卷积层，实际 {len(convs)} 个"
    return f"卷积层数量 = {len(convs)} ✓"


def c2_第一个卷积层输入是3通道():
    m = mod.build_cnn()
    convs = [layer for layer in m if isinstance(layer, nn.Conv2d)]
    assert convs[0].in_channels == 3, \
        f"彩色图 RGB 是 3 通道，第一层 in_channels 应为 3，实际 {convs[0].in_channels}"
    return f"第一个卷积层 in_channels = {convs[0].in_channels} ✓"


def c3_全连接层输入维度正确():
    m = mod.build_cnn()
    linears = [layer for layer in m if isinstance(layer, nn.Linear)]
    assert linears[-1].in_features == 2048, \
        f"展平后应为 128×4×4 = 2048，实际 {linears[-1].in_features}（检查三次池化后尺寸对不对）"
    assert linears[-1].out_features == 10, f"输出应为 10 类，实际 {linears[-1].out_features}"
    return f"全连接层 in_features = {linears[-1].in_features}、out_features = {linears[-1].out_features} ✓"


def c4_训练后测试准确率达标():
    train_loader, test_loader = mod.load_cifar(batch_size=128)
    m = mod.build_cnn().to(mod.device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    for _ in range(2):
        tr_loss, tr_acc = mod.train_epoch(m, train_loader, opt, loss_fn)
    te_loss, te_acc = mod.evaluate(m, test_loader, loss_fn)
    print(f"    2 轮训练：train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")
    assert te_acc > 0.55, \
        f"测试准确率 {te_acc:.4f} 偏低（三层 CNN 2 轮应 > 0.55），检查卷积结构"
    return f"训练 {tr_acc:.4f} / 测试 {te_acc:.4f} ✓"


if __name__ == "__main__":
    checks = [c1_模型含三个卷积层, c2_第一个卷积层输入是3通道,
              c3_全连接层输入维度正确, c4_训练后测试准确率达标]
    ok = 0
    for fn in checks:
        try:
            msg = fn()
            print(f"[通过] {fn.__name__}: {msg}")
            ok += 1
        except AssertionError as e:
            print(f"[失败] {fn.__name__}: {e}")
        except Exception as e:
            print(f"[出错] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n===== {ok}/{len(checks)} 项通过 =====")
