# -*- coding: utf-8 -*-
r"""
Day07 nn+optim 线性回归 · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day07_自检.py
"""
import importlib.util
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day07_线性回归.py")

print("=" * 50)
print("Day07 nn+optim 线性回归 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("lr7", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except SyntaxError as e:
    print(f"[加载失败] 语法错误：第 {e.lineno} 行 -> {e.msg}")
    raise SystemExit(1)
except Exception as e:
    print(f"[加载失败] {type(e).__name__}: {e}")
    raise SystemExit(1)


class NotDone(Exception):
    pass


def call(fn, *a, **k):
    r = fn(*a, **k)
    if r is None:
        raise NotDone()
    return r


results = []


def run(name, func):
    try:
        func()
        results.append((name, "PASS"))
    except NotDone:
        results.append((name, "未完成"))
    except AssertionError as e:
        results.append((name, f"FAIL  {e}" if str(e) else "FAIL"))
    except Exception as e:
        results.append((name, f"ERROR {type(e).__name__}: {e}"))


def c1():
    """数据形状"""
    X, y = mod.make_data(100, 0)
    assert X.shape == (100, 1) and y.shape == (100, 1), f"形状不对 {X.shape} {y.shape}"


def c2():
    """train 返回 nn.Linear 模型，参数形状正确"""
    X, y = mod.make_data(10, 0)
    model, losses = call(mod.train, X, y)
    assert isinstance(model, nn.Linear), f"train 应返回 nn.Linear，实际 {type(model)}"
    assert model.weight.shape == (1, 1) and model.bias.shape == (1,), \
        f"模型参数形状不对 {model.weight.shape} {model.bias.shape}"


def c3():
    """训练后 w≈3, b≈2"""
    X, y = mod.make_data(200, 0)
    model, losses = call(mod.train, X, y)
    assert abs(model.weight.item() - 3) < 0.1, f"w 应接近 3，实际 {model.weight.item():.3f}"
    assert abs(model.bias.item() - 2) < 0.1, f"b 应接近 2，实际 {model.bias.item():.3f}"


def c4():
    """loss 下降"""
    X, y = mod.make_data(200, 0)
    model, losses = call(mod.train, X, y)
    assert losses[-1] < losses[0], "loss 应下降"
    assert losses[-1] < 0.05, f"最终 loss 应 <0.05，实际 {losses[-1]:.4f}"


for i, fn in enumerate([c1, c2, c3, c4], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 4")
