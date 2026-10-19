# -*- coding: utf-8 -*-
r"""
Day12 test 集 + 交叉验证 · 自检脚本
用法：
    conda activate ai
    python Day12_自检.py
"""
import importlib.util
import os
import statistics

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day12_评估方法.py")

print("=" * 50)
print("Day12 test 集 + 交叉验证 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("fit12", TARGET)
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


X, y = mod.make_data(400, seed=0)


def c1():
    """split_three 三块数量正确，且总和 = 总数"""
    r = call(mod.split_three, X, y, 0.6, 0.2, 7)
    if any(v is None for v in r):
        raise NotDone()
    X_tr, y_tr, X_va, y_va, X_te, y_te = r
    n = len(y)
    assert len(y_tr) == int(n * 0.6), f"train 数量错：{len(y_tr)}"
    assert len(y_va) == int(n * 0.2), f"val 数量错：{len(y_va)}"
    assert len(y_te) == n - int(n * 0.6) - int(n * 0.2), f"test 数量错：{len(y_te)}"
    assert len(X_tr) == len(y_tr) and len(X_va) == len(y_va) and len(X_te) == len(y_te)


def c2():
    """split_three 打乱了：每块都含两类（顺序切会全是一类）"""
    r = call(mod.split_three, X, y, 0.6, 0.2, 7)
    if any(v is None for v in r):
        raise NotDone()
    _, y_tr, _, y_va, _, y_te = r
    for name, yy in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        uniq = set(yy.flatten().tolist())
        assert uniq == {0.0, 1.0}, f"{name} 集不是两类都有：{uniq}（可能忘了打乱）"


def c3():
    """kfold_scores 返回恰好 k 个分数"""
    r = call(mod.kfold_scores, X, y, k=5)
    if r is None:
        raise NotDone()
    assert isinstance(r, list), "应返回列表"
    assert len(r) == 5, f"应返回 5 个分数，实际 {len(r)}"
    assert all(0.0 <= s <= 1.0 for s in r), f"分数应在 [0,1]，实际 {r}"


def c4():
    """k 折平均验证准确率合理（模型确实学到了）"""
    r = call(mod.kfold_scores, X, y, k=5)
    if r is None:
        raise NotDone()
    mean = statistics.mean(r)
    assert mean > 0.80, f"k 折平均应 > 0.80，实际 {mean:.3f}（检查训练/打分是否接对了）"


for i, fn in enumerate([c1, c2, c3, c4], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 4")
