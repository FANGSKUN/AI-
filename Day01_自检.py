# -*- coding: utf-8 -*-
r"""
Day01 摸底测试 · 自动检查脚本
================================
用法：
  1. 先把 Day01_摸底测试.py 里的函数尽量填好
  2. 在终端里运行：
         conda activate ai
         cd "C:\Users\29322\Desktop\AI自学"
         python Day01_自检.py
  3. 看每题结果是 PASS / FAIL / 未完成

说明：本脚本只判断对错，不会给出答案。答案今晚随学习报告一起给你。
"""
import importlib.util
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Day01_摸底测试.py")

print("=" * 50)
print("Day01 摸底测试 自检")
print("=" * 50)

try:
    spec = importlib.util.spec_from_file_location("modi", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except SyntaxError as e:
    print(f"[加载失败] 你的文件有语法错误：第 {e.lineno} 行 -> {e.msg}")
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
    except AttributeError as e:
        results.append((name, f"未完成 ({e})"))
    except Exception as e:
        results.append((name, f"ERROR {type(e).__name__}: {e}"))


def q1():
    assert call(mod.q1, [1, 2, 3, 4]) == 20, "样例 [1,2,3,4] 应为 20"
    assert call(mod.q1, [2, 4, 6]) == 56, "样例 [2,4,6] 应为 56"


def q2():
    rev, cnt = call(mod.q2, "hello")
    assert rev == "olleh", "反转结果不对"
    assert dict(cnt) == {"h": 1, "e": 1, "l": 2, "o": 1}, "字符计数不对"


def q3():
    assert call(mod.q3) == [3, 5, 7], "切片结果应为 [3,5,7]"


def q4():
    assert call(mod.q4, {"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 5, "c": 4}, "合并结果不对"


def q5():
    assert tuple(call(mod.q5, 3, 1, 2)) == (1, 3), "多参数结果不对"
    assert tuple(call(mod.q5)) == (None, None), "无参数应返回 (None, None)"


def q6():
    r = list(call(mod.q6, 5))
    assert r == [1, 1, 2, 3, 5], f"应为 [1,1,2,3,5]，你的是 {r}"


def q7():
    assert call(mod.q7, [1, 3, 5, 7, 9], 5) == 2, "找到时应返回索引 2"
    assert call(mod.q7, [1, 3, 5], 4) == -1, "找不到应返回 -1"


def q8():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("a a a b b c")
        top = list(call(mod.q8, p))[0]
        assert tuple(top) == ("a", 3), f"最高频应为 ('a',3)，你的是 {top}"


def q9():
    with tempfile.TemporaryDirectory() as d:
        for n in ["a.txt", "b.txt", "c.md"]:
            open(os.path.join(d, n), "w").close()
        names = sorted(os.path.basename(str(x)) for x in call(mod.q9, d))
        assert names == ["a.txt", "b.txt"], f"应只返回两个 txt，你的是 {names}"


def q10():
    s = mod.Stack()
    assert s.is_empty() is True, "初始应为空"
    s.push(1)
    s.push(2)
    assert s.peek() == 2, "栈顶应为 2"
    assert s.pop() == 2, "pop 应返回 2"
    assert s.is_empty() is False
    s.pop()
    try:
        s.pop()
    except IndexError:
        return
    raise AssertionError("空栈 pop 应抛出 IndexError")


def q11():
    assert call(mod.q11, 6, 2) == 3.0, "6/2 应为 3.0"
    assert call(mod.q11, 1, 0) == "除零错误", "除零应返回 '除零错误'"


def q12():
    assert call(mod.q12, "1+2*3-4") == 3, "1+2*3-4 应为 3"
    assert call(mod.q12, "2*3+4") == 10, "2*3+4 应为 10"


for i, fn in enumerate([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12], start=1):
    run(f"q{i}", fn)

print()
for name, status in results:
    print(f"  {name:>4} : {status}")

ok = sum(1 for _, s in results if s == "PASS")
print()
print(f"通过 {ok} / 12")
print()
print("提示：FAIL/ERROR 说明逻辑有误；未完成说明还没写或返回了 None。")
print("先别纠结正确答案，把每题感受记下来，今晚汇报给我。")
