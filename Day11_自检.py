# -*- coding: utf-8 -*-
r"""
Day11 正则化 + 早停 · 自检脚本
用法：
    conda activate ai
    python Day11_自检.py
"""
import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day11_正则化早停.py")

print("=" * 50)
print("Day11 正则化 + 早停 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("fit11", TARGET)
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


def weight_norm(model):
    return sum((p ** 2).sum().item() for p in model.parameters())


def c1():
    """weight_decay 让大模型的权重更小（L2 惩罚生效）"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.25, 7)
    m0, *_ = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64)
    m1, *_ = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64, weight_decay=0.1)
    w0, w1 = weight_norm(m0), weight_norm(m1)
    assert w1 < w0, f"weight_decay 应减小权重：{w1:.3f} vs {w0:.3f}"


def c2():
    """weight_decay 减小过拟合 gap"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.25, 7)
    m0, b_tr, b_va, _ = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64)
    m1, w_tr, w_va, _ = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64, weight_decay=0.1)
    gap0 = b_tr[-1] - b_va[-1]
    gap1 = w_tr[-1] - w_va[-1]
    assert gap1 < gap0, f"weight_decay 应减小 gap：{gap1:+.3f} vs {gap0:+.3f}"


def c3():
    """early_stop 真的提前停了（没跑满 5000 轮）"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.25, 7)
    m, tr, va, ep = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64,
                         early_stop=True, patience=200)
    assert ep < 5000, f"早停应提前停，实际第 {ep} 轮"


def c4():
    """early_stop 回退了：返回模型的验证准确率不差于最后一步"""
    X, y = mod.make_data(400, 0)
    X_tr, y_tr, X_va, y_va = mod.split(X, y, 0.25, 7)
    m, tr, va, ep = call(mod.train_mlp, X_tr, y_tr, X_va, y_va, hidden=64,
                         early_stop=True, patience=200)
    assert len(tr) > 0 and len(va) > 0, "准确率曲线为空"
    # 早停后的模型在验证集上应该不错（>= 0.85）
    val = mod.accuracy(m, X_va, y_va)
    assert val >= 0.85, f"早停后验证准确率应 >=0.85，实际 {val:.3f}"


for i, fn in enumerate([c1, c2, c3, c4], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 4")
