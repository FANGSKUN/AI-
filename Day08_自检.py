# -*- coding: utf-8 -*-
r"""
Day08 逻辑回归 · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day08_自检.py
"""
import importlib.util
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day08_逻辑回归.py")

print("=" * 50)
print("Day08 逻辑回归 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("lr8", TARGET)
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
    assert X.shape == (200, 2) and y.shape == (200, 1), f"形状不对 {X.shape} {y.shape}"


def c2():
    """train 返回 nn.Linear，参数形状正确"""
    X, y = mod.make_data(20, 0)
    model, losses = call(mod.train, X, y)
    assert isinstance(model, nn.Linear), f"train 应返回 nn.Linear，实际 {type(model)}"
    assert model.weight.shape == (1, 2) and model.bias.shape == (1,), \
        f"参数形状不对 {model.weight.shape} {model.bias.shape}"
    assert len(losses) > 0, "losses 为空，训练循环没执行"


def c3():
    """predict 输出 0/1，形状正确"""
    X, y = mod.make_data(20, 0)
    model, _ = call(mod.train, X, y)
    pred = call(mod.predict, model, X)
    assert pred.shape == (40, 1), f"predict 形状不对 {pred.shape}"
    assert set(pred.unique().tolist()) <= {0.0, 1.0}, f"预测值应只有 0/1，实际 {pred.unique().tolist()}"


def c4():
    """训练后准确率高（>0.9），且 loss 下降"""
    X, y = mod.make_data(100, 0)
    model, losses = call(mod.train, X, y)
    acc = call(mod.accuracy, model, X, y)
    assert acc > 0.9, f"准确率应 >0.9，实际 {acc:.3f}"
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
