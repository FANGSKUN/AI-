# -*- coding: utf-8 -*-
"""
Day14 自检：CNN 卷积神经网络
运行：
    conda activate ai
    python Day14_自检.py
"""
import torch
import torch.nn as nn

import Day14_CNN卷积 as mod


def c1_模型包含卷积层():
    m = mod.build_cnn()
    convs = [layer for layer in m if isinstance(layer, nn.Conv2d)]
    assert len(convs) >= 2, f"CNN 至少要有 2 个卷积层，你的模型里只有 {len(convs)} 个"
    return f"卷积层数量 = {len(convs)} ✓"


def c2_第一个卷积层输入是1通道():
    m = mod.build_cnn()
    convs = [layer for layer in m if isinstance(layer, nn.Conv2d)]
    first = convs[0]
    assert first.in_channels == 1, f"第一层卷积的输入应该是 1 通道（灰度图），实际是 {first.in_channels}"
    return f"第一个卷积层 in_channels = {first.in_channels} ✓"


def c3_全连接层输入维度正确():
    m = mod.build_cnn()
    linears = [layer for layer in m if isinstance(layer, nn.Linear)]
    last = linears[-1]
    assert last.in_features == 1568, \
        f"展平后应该是 32×7×7 = 1568，你的 Linear 输入是 {last.in_features}（检查卷积/池化后尺寸对不对）"
    assert last.out_features == 10, f"输出应该是 10 类，实际是 {last.out_features}"
    return f"全连接层 in_features = {last.in_features}、out_features = {last.out_features} ✓"


def c4_训练后测试准确率达标():
    train_loader, test_loader = mod.load_mnist(batch_size=128)
    m = mod.build_cnn()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    tr_loss, tr_acc = mod.train_epoch(m, train_loader, opt, loss_fn)
    te_loss, te_acc = mod.evaluate(m, test_loader, loss_fn)
    print(f"    1 轮训练：train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")
    assert te_acc > 0.97, f"测试准确率 {te_acc:.4f} 偏低（CNN 应 > 0.97），检查卷积结构"
    return f"训练 {tr_acc:.4f} / 测试 {te_acc:.4f} ✓"


if __name__ == "__main__":
    checks = [c1_模型包含卷积层, c2_第一个卷积层输入是1通道,
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
