# -*- coding: utf-8 -*-
r"""
Day04 KNN · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day04_自检.py
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day04_KNN.py")

print("=" * 50)
print("Day04 KNN 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("knn", TARGET)
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


X_all, y_all = mod.load_data()


def c1():
    """距离矩阵形状与数值"""
    Xte = X_all[:5]
    Xtr = X_all[5:15]
    D = np.asarray(call(mod.euclidean_distances, Xte, Xtr))
    assert D.shape == (5, 10), f"形状应为 (5,10)，实际 {D.shape}"
    # 参考值：按定义逐个算
    ref = np.sqrt(((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
    assert np.allclose(D, ref, atol=1e-4), "距离数值不对"


def c2():
    """k=1 的准确率应很高"""
    Xtr, ytr, Xte, yte = mod.split(X_all, y_all)
    pred = call(mod.knn_predict, Xte, Xtr, ytr, 1)
    pred = np.asarray(pred)
    assert pred.shape == (len(Xte),), f"预测形状应为 ({len(Xte)},)，实际 {pred.shape}"
    acc = mod.accuracy(pred, yte)
    assert acc > 0.95, f"k=1 准确率应 >0.95，实际 {acc:.3f}"


def c3():
    """k=3 的准确率"""
    Xtr, ytr, Xte, yte = mod.split(X_all, y_all)
    pred = np.asarray(call(mod.knn_predict, Xte, Xtr, ytr, 3))
    acc = mod.accuracy(pred, yte)
    assert acc > 0.90, f"k=3 准确率应 >0.90，实际 {acc:.3f}"


for i, fn in enumerate([c1, c2, c3], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 3")
