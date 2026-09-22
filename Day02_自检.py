# -*- coding: utf-8 -*-
r"""
Day02 NumPy 练习 · 自检脚本
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day02_自检.py
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day02_NumPy练习.py")

print("=" * 50)
print("Day02 NumPy 练习 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("ex", TARGET)
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


def call(fn, *a):
    r = fn(*a)
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


A = np.arange(20).reshape(4, 5)   # Part B 用
B = np.arange(15).reshape(5, 3)   # Part C/E 用
C = np.arange(12).reshape(3, 4)   # Part D 用


def p1():
    r = call(mod.p1)
    assert r.shape == (3, 4), f"shape 应为 (3,4)，实际 {r.shape}"
    assert r.dtype == np.float32, f"dtype 应为 float32，实际 {r.dtype}"


def p2():
    r = call(mod.p2)
    assert r.shape == (2, 5), f"shape 应为 (2,5)，实际 {r.shape}"
    assert r[0, 0] == 0 and r[1, 0] == 5, "内容不对"


def p3():
    r = call(mod.p3, A)
    assert tuple(r[0]) == (4, 5), "shape 不对"
    assert r[1] == 2, "ndim 不对"
    assert r[2] == A.dtype, "dtype 不对"
    assert r[3] == 20, "size 不对"


def p4():
    assert np.array_equal(call(mod.p4, A), A[1]), "第2行取错了"


def p5():
    r = call(mod.p5, A)
    assert r.shape == (4,), f"应是一维 (4,)，实际 {r.shape}"
    assert np.array_equal(r, A[:, 0]), "第1列取错了"


def p6():
    assert np.array_equal(call(mod.p6, A), A[2:, 3:]), "右下角 2x2 取错了"


def p7():
    assert np.array_equal(np.sort(call(mod.p7, A)), A[A > 10]), "筛选结果不对"


def p8():
    r = call(mod.p8)
    assert r.shape == (3, 4), f"shape 应为 (3,4)，实际 {r.shape}"
    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    assert np.array_equal(r, a + b), "数值不对"


def p9():
    r = call(mod.p9, B)
    assert r.shape == (5, 3), "shape 不对"
    assert np.allclose(r.mean(axis=0), 0), "每列减完均值后，列均值应约为 0"


def p10():
    s0, s1 = call(mod.p10, C)
    assert np.array_equal(s0, C.sum(axis=0)), "axis=0 不对"
    assert np.array_equal(s1, C.sum(axis=1)), "axis=1 不对"


def p11():
    r = call(mod.p11, C)
    assert r.shape == (3,), f"应是一维 (3,)，实际 {r.shape}"
    assert np.allclose(r, C.mean(axis=1)), "每行均值算错了"


def p12():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = call(mod.p12, x)
    assert abs(r.mean()) < 1e-9, "标准化后均值应为 0"
    assert abs(r.std() - 1) < 1e-9, "标准化后标准差应为 1"


def p13():
    r = call(mod.p13, B)
    assert r.shape == (5, 3), "shape 不对"
    assert np.allclose(r.mean(axis=1), 0), "每行减完均值后，行均值应约为 0"


for i, fn in enumerate(
    [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13], start=1
):
    run(f"p{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 13")
print()
print("卡住的题记下来，汇报时告诉我。")
