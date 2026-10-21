# -*- coding: utf-8 -*-
"""
Day16 自检：数据增强（Data Augmentation）
运行：
    conda activate ai
    python Day16_自检.py
"""
import torch
import torch.nn as nn
from torchvision import transforms

import Day16_数据增强 as mod


def get_transforms():
    train_loader, test_loader = mod.load_cifar(batch_size=64)
    return train_loader.dataset.transform, test_loader.dataset.transform


def c1_训练集含随机水平翻转():
    train_t, _ = get_transforms()
    has_flip = any(isinstance(t, transforms.RandomHorizontalFlip)
                   for t in train_t.transforms)
    assert has_flip, "训练集 transform 里缺少 RandomHorizontalFlip（随机水平翻转）"
    return "训练集含 RandomHorizontalFlip ✓"


def c2_训练集含随机裁剪且参数正确():
    train_t, _ = get_transforms()
    crops = [t for t in train_t.transforms if isinstance(t, transforms.RandomCrop)]
    assert crops, "训练集 transform 里缺少 RandomCrop（随机裁剪）"
    crop = crops[0]
    assert crop.size == (32, 32), f"RandomCrop 尺寸应为 32，实际 {crop.size}"
    assert crop.padding == 4, \
        f"RandomCrop padding 应为 4（先补边再裁），实际 {crop.padding}"
    return "训练集含 RandomCrop(32, padding=4) ✓"


def c3_测试集不增强():
    _, test_t = get_transforms()
    has_aug = any(isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomCrop))
                  for t in test_t.transforms)
    assert not has_aug, "测试集不能增强（要保持原样评估，否则分数是假的）"
    return "测试集无增强（保持原样）✓"


def c4_训练后测试准确率达标():
    train_loader, test_loader = mod.load_cifar(batch_size=128)
    m = mod.build_cnn().to(mod.device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    for _ in range(15):
        tr_loss, tr_acc = mod.train_epoch(m, train_loader, opt, loss_fn)
    te_loss, te_acc = mod.evaluate(m, test_loader, loss_fn)
    print(f"    15 轮训练（有增强）：train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")
    assert te_acc > 0.75, \
        f"有数据增强后 test_acc {te_acc:.4f} 应 > 0.75（增强版 15 轮约 0.76；无增强 5 轮只有 0.70），检查增强有没有生效"
    return f"增强后训练 {tr_acc:.4f} / 测试 {te_acc:.4f} ✓"


if __name__ == "__main__":
    checks = [c1_训练集含随机水平翻转, c2_训练集含随机裁剪且参数正确,
              c3_测试集不增强, c4_训练后测试准确率达标]
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
