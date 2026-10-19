# -*- coding: utf-8 -*-
"""
Day13 自检：MNIST 多分类
运行：
    conda activate ai
    python Day13_自检.py
"""
import torch
import torch.nn as nn
import importlib

import Day13_MNIST多分类 as mod


def c1_输出层是10个神经元():
    m = mod.build_model(hidden=32)
    linears = [layer for layer in m if isinstance(layer, nn.Linear)]
    last = linears[-1]
    assert last.out_features == 10, f"输出层应该是 10 个神经元（多分类 10 类），实际是 {last.out_features}"
    return f"输出层 out_features = {last.out_features} ✓"


def c2_损失是CrossEntropyLoss():
    loss = mod.build_loss()
    assert isinstance(loss, nn.CrossEntropyLoss), \
        f"多分类应该用 nn.CrossEntropyLoss()，你用的是 {type(loss).__name__}"
    return f"损失函数是 {type(loss).__name__} ✓"


def c3_predict返回整数标签():
    m = mod.build_model(hidden=32)
    x = torch.randn(16, 1, 28, 28)
    p = mod.predict(m, x)
    assert p is not None, "predict 还没实现（还是 None）"
    assert tuple(p.shape) == (16,), f"predict 输出形状应该是 (16,)，实际是 {tuple(p.shape)}"
    assert p.dtype in (torch.int64, torch.long), f"预测结果应该是整数标签，实际 dtype {p.dtype}"
    assert (p >= 0).all() and (p <= 9).all(), f"类别应该在 0~9 之间，实际范围 {p.min().item()}~{p.max().item()}"
    return f"predict 输出形状 {tuple(p.shape)}、整数、值域 0~9 ✓"


def c4_训练后测试准确率达标():
    # 用你自己的 build_model + build_loss + train_epoch + evaluate 完整跑一遍
    train_loader, test_loader = mod.load_mnist(batch_size=128)
    m = mod.build_model(hidden=128)
    loss_fn = mod.build_loss()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    tr_loss, tr_acc = mod.train_epoch(m, train_loader, opt, loss_fn)
    te_loss, te_acc = mod.evaluate(m, test_loader, loss_fn)
    print(f"    1 轮训练：train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")
    assert tr_acc > 0.5, f"训练 1 轮准确率才 {tr_acc:.4f}，说明训练没接对（检查损失函数/输出层）"
    assert te_acc > 0.85, f"测试准确率 {te_acc:.4f} 偏低（应 > 0.85），检查预测或训练"
    return f"训练 {tr_acc:.4f} / 测试 {te_acc:.4f} ✓"


if __name__ == "__main__":
    checks = [c1_输出层是10个神经元, c2_损失是CrossEntropyLoss,
              c3_predict返回整数标签, c4_训练后测试准确率达标]
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
