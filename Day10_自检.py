# -*- coding: utf-8 -*-
r"""
Day10 过拟合 · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day10_自检.py
"""
import importlib.util
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day10_过拟合.py")

print("=" * 50)
print("Day10 过拟合 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("fit10", TARGET)
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
    """split 切分大小正确，且 train/val 不重叠"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = call(mod.split, X, y, 0.25, 7)
    assert X_tr.shape[0] == 100 and X_va.shape[0] == 300, \
        f"切分大小不对 {X_tr.shape[0]} {X_va.shape[0]}"
    assert X_tr.shape[1] == 2 and y_tr.shape[1] == 1, "形状不对"


def c2():
    """train_mlp 返回模型，且记录了准确率曲线"""
    X, y = mod.make_data(200, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.5, 7)
    model, tr, va = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=16, epochs=1000)
    assert isinstance(model, nn.Sequential), f"train_mlp 应返回 nn.Sequential，实际 {type(model)}"
    assert len(tr) > 0 and len(va) > 0, "准确率曲线为空，检查 TODO 2 的记录逻辑"


def c3():
    """大模型在训练集上过拟合：训练准确率接近 1"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.25, 7)
    model, tr, va = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64)
    assert tr[-1] > 0.95, f"大模型训练准确率应 >0.95，实际 {tr[-1]:.3f}"


def c4():
    """过拟合的表现：训练明显高于验证（存在 gap）"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.25, 7)
    model, tr, va = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64)
    gap = tr[-1] - va[-1]
    assert gap > 0.02, f"应观察到训练>验证的 gap，实际 {gap:+.3f}"


for i, fn in enumerate([c1, c2, c3, c4], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 4")
