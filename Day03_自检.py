# -*- coding: utf-8 -*-
r"""
Day03 线性回归 · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day03_自检.py
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day03_线性回归.py")

print("=" * 50)
print("Day03 线性回归 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("lr", TARGET)
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
    """MSE 公式是否正确"""
    y_pred = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[1.0], [1.0], [1.0]])
    # 误差 0,1,2 -> 平方 0,1,4 -> 均值 5/3
    r = call(mod.mse_loss, y_pred, y)
    assert abs(float(r) - 5 / 3) < 1e-9, f"MSE 应为 5/3，你的是 {r}"


def c2():
    """梯度是否正确（用数值梯度对照）"""
    X = np.array([[0.2], [0.5], [0.8]])
    y = np.array([[1.0], [2.0], [3.0]])
    w = np.array([[0.3]])
    b = 0.1

    dw, db = call(mod.gradients, X, y, w, b)

    # 数值梯度
    eps = 1e-6
    def loss(w_, b_):
        return float(mod.mse_loss(mod.predict(X, w_, b_), y))
    dw_num = (loss(w + eps, b) - loss(w - eps, b)) / (2 * eps)
    db_num = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)

    assert abs(float(np.asarray(dw).ravel()[0]) - dw_num) < 1e-3, \
        f"dw 不对：解析 {float(np.asarray(dw).ravel()[0]):.6f} vs 数值 {dw_num:.6f}"
    assert abs(float(db) - db_num) < 1e-3, \
        f"db 不对：解析 {float(db):.6f} vs 数值 {db_num:.6f}"


def c3():
    """训练后能否拟合出 w≈3, b≈2"""
    X, y = mod.make_data(200, 0)
    w, b, losses = call(mod.train, X, y)
    w_val = float(np.asarray(w).ravel()[0])
    assert abs(w_val - 3) < 0.1, f"w 应接近 3，你的是 {w_val:.3f}"
    assert abs(float(b) - 2) < 0.1, f"b 应接近 2，你的是 {float(b):.3f}"


def c4():
    """loss 是否随训练下降"""
    X, y = mod.make_data(200, 0)
    w, b, losses = call(mod.train, X, y)
    assert losses[-1] < losses[0], "最终 loss 应小于初始 loss"
    assert losses[-1] < 0.05, f"最终 loss 应很小(<0.05)，你的是 {losses[-1]:.4f}"


for i, fn in enumerate([c1, c2, c3, c4], start=1):
    run(f"c{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 4")
print()
print("全部通过后，运行 Day03_线性回归.py 看拟合图和 loss 曲线。")
