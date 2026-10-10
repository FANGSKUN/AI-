# -*- coding: utf-8 -*-
r"""
Day06 PyTorch 线性回归 · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day06_自检.py
"""
import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day06_线性回归.py")

print("=" * 50)
print("Day06 PyTorch 线性回归 自检")
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
    """数据形状"""
    X, y = mod.make_data(100, 0)
    assert X.shape == (100, 1) and y.shape == (100, 1), f"形状不对 {X.shape} {y.shape}"


def c2():
    """backward 后梯度是否正确（对照手推公式）"""
    X, y = mod.make_data(20, 0)
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        err = X @ w + b - y
        dw_manual = 2 * (err * X).mean(0, keepdim=True)   # (1,1)
        db_manual = 2 * err.mean()                          # 标量

    assert w.grad is not None, "w.grad 是 None，检查 requires_grad 或 backward"
    assert torch.allclose(w.grad, dw_manual, atol=1e-5), f"dw 不对：{w.grad} vs {dw_manual}"
    assert torch.allclose(b.grad, db_manual, atol=1e-5), f"db 不对：{b.grad} vs {db_manual}"


def c3():
    """训练后 w≈3, b≈2"""
    X, y = mod.make_data(200, 0)
    w, b, losses = call(mod.train, X, y)
    assert abs(w.item() - 3) < 0.1, f"w 应接近 3，实际 {w.item():.3f}"
    assert abs(b.item() - 2) < 0.1, f"b 应接近 2，实际 {b.item():.3f}"


def c4():
    """loss 下降"""
    X, y = mod.make_data(200, 0)
    w, b, losses = call(mod.train, X, y)
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
