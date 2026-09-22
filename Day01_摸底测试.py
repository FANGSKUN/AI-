# -*- coding: utf-8 -*-
"""
Day 01 摸底测试
================
规则：
  1. 不查资料、不查答案、不问 AI。
  2. 限时 60 分钟。
  3. 每题直接在对应函数里写代码。
  4. 不会的题不要空着：写一行注释 "# 卡住：……"，然后跳过。
  5. 做完后，在最下方「自评区」填写每题难度和总时长。

运行方式（可选，用来验证你的答案）：
    conda activate ai
    python Day01_摸底测试.py
"""

# ============================================================
# Part 1  基础语法与数据结构
# ============================================================
from collections import Counter
from pathlib import Path
import time
def q1(nums):
    """给定数字列表 nums，用一行代码返回所有偶数的平方和。
    例：[1, 2, 3, 4] -> 2**2 + 4**2 = 20
    """
    return sum(x**2 for x in nums if x % 2 ==0)


def q2(s):
    """给定字符串 s，返回一个元组 (反转串, 字符计数字典)。
    例：s = "hello" -> ("olleh", {'h':1, 'e':1, 'l':2, 'o':1})
    """
    res = (s[::-1],dict(Counter(s)))
    return res


def q3():
    """给定 a = [1,2,3,4,5,6,7,8,9,10]，
    用切片取出索引 2 到 6（含）之间、步长为 2 的元素。
    期望结果：[3, 5, 7]
    """
    a=[1,2,3,4,5,6,7,8,9,10]
    b=a[2:7:2]
    return b


def q4(d1, d2):
    """合并字典 d1、d2，相同 key 的值相加。
    例：d1={'a':1,'b':2}, d2={'b':3,'c':4} -> {'a':1,'b':5,'c':4}
    """
    d = d1.copy()
    for i,j in d2.items():
        d[i] = d.get(i,0) + j
    return d


# ============================================================
# Part 2  函数与进阶
# ============================================================

def q5(*args):
    """接收任意个数字，返回 (最小值, 最大值)；无参数时返回 (None, None)。
    """
    return (min(args),max(args)) if args else (None,None)


def q6(n):
    """写一个生成器函数，惰性产出斐波那契数列前 n 项。
    q6(5) -> 1 1 2 3 5
    """
    a,b=1,1
    for _ in range(n):
        yield a
        a,b = b,a+b
    pass


def q7(sorted_list, target):
    """在已排序列表 sorted_list 中查找 target，返回索引；找不到返回 -1。
    要求：时间复杂度 O(log n)（二分查找）。
    """
    left,right = 0, len(sorted_list)-1
    while left <= right:
        mid = (left+right)//2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            left = mid+1
        else:
            right = mid -1
    return -1


# ============================================================
# Part 3  文件与标准库
# ============================================================
def q8(filepath):
    """读取文本文件 filepath，返回出现次数最多的前 10 个单词及次数。
    返回格式：[('the', 30), ('a', 20), ...]
    提示：可用 collections.Counter；暂不要求处理标点。
    """
    with open(filepath, 'r', encoding = 'utf-8') as f:
        words = f.read().split()
    return Counter(words).most_common(10)

def q9(dirpath):
    """用 pathlib 遍历目录 dirpath，返回所有 .txt 文件的路径列表。
    """
    p = Path(dirpath)
    for f in p.glob("*.txt"):
        yield f
    pass


# ============================================================
# Part 4  面向对象与异常
# ============================================================

class Stack:
    """实现一个栈：
        push(x)
        pop()       空栈时抛出 IndexError
        peek()      返回栈顶但不出栈
        is_empty()  -> bool
    """
    # 在这里实现
    def __init__(self):
        self.items = []
    def push(self,x):
        self.items.append(x)
    def pop(self):
        if self.is_empty():
            raise IndexError("IndexError")
        return self.items.pop()
    def is_empty(self):
        return len(self.items) == 0
    def peek(self):
        if self.is_empty():
            raise IndexError("IndexError")
        return self.items[-1]
    pass


def q11(a, b):
    """返回 a / b；若 b == 0，捕获异常并返回字符串 "除零错误"。
    """
    try:
        return a/b
    except ZeroDivisionError:
        return "除零错误"


# ============================================================
# Part 5  综合（选做，做不出没关系）
# ============================================================

def q12(expr):
    # 1. 分词
    tokens = []
    num = ''
    for ch in expr:
        if ch.isdigit():
            num += ch
        else:
            tokens.append(int(num))
            tokens.append(ch)
            num = ''
    tokens.append(int(num))   # 循环外

    # 2. 用标记法处理乘法，边读边合并
    n = Stack()
    op = Stack()
    pending_mul = False
    for x in tokens:
        if isinstance(x, int):
            if pending_mul:
                n.push(n.pop() * x)
                pending_mul = False
            else:
                n.push(x)
        elif x == '*':
            pending_mul = True
        else:                 # + 或 -
            op.push(x)

    # 3. 从左到右算加减（用下标，别弹栈）
    result = n.items[0]
    for i, o in enumerate(op.items):
        if o == '+':
            result += n.items[i + 1]
        else:
            result -= n.items[i + 1]
    return result
    pass


# ============================================================
# 自评区（做完后填写）
# ============================================================
# 难度感受（简单 / 一般 / 难）：
#   q1:简单      q2:一般      q3: 简单    q4:一般
#   q5: 难    q6:   一般   q7: 简单   q8:难
#   q9: 一般   q10:难     q11:难   q12:难
#
# 最卡的地方是：q12
#
# 花费总时长：1h

if __name__ == "__main__":
    # 你可以在这里调用上面的函数，验证自己的答案（可选）


    for i in range(101):  # 添加进度条图形和百分比
        bar = '[' + '=' * (i // 2) + ' ' * (50 - i // 2) + ']'
        print(f"\r{bar} {i:3}%", end='', flush=True)
        time.sleep(0.05)
    print()
    pass
