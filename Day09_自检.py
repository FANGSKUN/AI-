# -*- coding: utf-8 -*-
r"""
Day09 MLP · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day09_自检.py
"""
import importlib.util
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day09_MLP.py")

print("=" * 50)
print("Day09 MLP 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("mlp9", TARGET)
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
    """XOR 数据形状"""
    X, y = mod.make_xor(25, 0)
    assert X.shape == (100, 2) and y.shape == (100, 1), f"形状不对 {X.shape} {y.shape}"


def c2():
    """单层线性模型在 XOR 上学不动（准确率应 < 0.7）"""
    X, y = mod.make_xor(25, 0)
    model, losses = call(mod.train_linear, X, y)
    acc = call(mod.accuracy, model, X, y)
    assert acc < 0.7, f"线性模型应学不动（<0.7），实际 {acc:.3f}，检查数据是不是 XOR"


def c3():
    """MLP 能学会 XOR（准确率应 > 0.9）"""
    X, y = mod.make_xor(25, 0)
    model, losses = call(mod.train_mlp, X, y)
    acc = call(mod.accuracy, model, X, y)
    assert acc > 0.9, f"MLP 准确率应 >0.9，实际 {acc:.3f}"


def c4():
    """MLP loss 下降"""
    X, y = mod.make_xor(25, 0)
    model, losses = call(mod.train_mlp, X, y)
    assert losses[-1] < losses[0], "loss 应下降"
    assert losses[-1] < 0.1, f"最终 loss 应 <0.1，实际 {losses[-1]:.4f}"


for i, fn in enumerate([c1, c2, c3, c4], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 4")
