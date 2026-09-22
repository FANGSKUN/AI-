# Day01 摸底测试 · 参考答案

> 说明：答案追求 **Pythonic 写法**，不只是"能跑对"。对照看看和你的写法差在哪。

---

## Part 1 基础语法与数据结构

**q1** 偶数平方和
```python
def q1(nums):
    return sum(x ** 2 for x in nums if x % 2 == 0)
```
> 你写得对。要点：生成器表达式 + `sum`，一行搞定，不建中间列表。

**q2** 反转 + 字符计数
```python
from collections import Counter

def q2(s):
    return (s[::-1], dict(Counter(s)))
```
> 你的 `{c: s.count(c) for c in set(s)}` 结果对，但 `s.count` 每个字符都重扫一遍 `s`，是 O(n·k)。`Counter` 只扫一次，O(n)。

**q3** 切片
```python
def q3():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return a[2:7:2]      # [3, 5, 7]
```
> 切片**左闭右开**：要含索引 6，就得写 7。你原来的 `a[2:6:2]` 少了 7，靠 `append(a[6])` 补，说明这个概念没吃透。

**q4** 字典合并相加
```python
from collections import Counter

def q4(d1, d2):
    c = Counter(d1)
    c.update(d2)
    return dict(c)
```
> 你的循环写法也对。用 `Counter` 更简洁，且天然支持"相加"语义。

---

## Part 2 函数与进阶

**q5** 任意参数求 min/max
```python
def q5(*args):
    return (min(args), max(args)) if args else (None, None)
```
> 你多套了一层内层函数，完全没必要。`*args` 本身就是元组，直接 `min/max` 即可。

**q6** 斐波那契生成器
```python
def q6(n):
    a, b = 1, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
```
> 你写得完全正确，`a, b = b, a + b` 是标准写法。

**q7** 二分查找
```python
def q7(sorted_list, target):
    left, right = 0, len(sorted_list) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
> 你写得正确。

---

## Part 3 文件与标准库

**q8** 词频统计
```python
from collections import Counter

def q8(filepath):
    with open(filepath, encoding="utf-8") as f:
        words = f.read().split()
    return Counter(words).most_common(10)
```
> 你写得正确，`most_common` 用得好。

**q9** 遍历 txt
```python
from pathlib import Path

def q9(dirpath):
    return list(Path(dirpath).glob("*.txt"))
```
> 你用了 `yield`，返回的是**生成器**而不是文档要求的**列表**。二者都对，但要和文档一致。

---

## Part 4 面向对象与异常

**q10** 栈
```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def is_empty(self):
        return not self.items
```
> 你写得基本正确。`len(self.items) == 0` 可简写为 `not self.items`。

**q11** 除零异常
```python
def q11(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "除零错误"
```
> 你写得正确。

---

## Part 5 综合

**q12** 表达式求值（无 eval）

思路：先分词，再用"标记法"先合并乘法，最后从左到右算加减。

```python
def q12(expr):
    # 1. 分词：把多位数"累积"出来
    tokens = []
    num = ""
    for ch in expr:
        if ch.isdigit():
            num += ch
        else:
            tokens.append(int(num))
            tokens.append(ch)
            num = ""
    tokens.append(int(num))          # 循环外，最后一个数字

    # 2. 先合并乘法
    nums = []
    ops = []
    pending_mul = False
    for x in tokens:
        if isinstance(x, int):
            if pending_mul:
                nums.append(nums.pop() * x)
                pending_mul = False
            else:
                nums.append(x)
        elif x == "*":
            pending_mul = True
        else:
            ops.append(x)

    # 3. 从左到右算加减
    result = nums[0]
    for i, op in enumerate(ops):
        result += nums[i + 1] if op == "+" else -nums[i + 1]
    return result
```

**关键点**
- 分词时数字要"累积"，且在**循环外**结算最后一个数字。
- `*` 不立即计算，而是打标记，**读下一个数字时**才合并（因为看到 `*` 时右操作数还没读进来）。
- 加减**从左到右**按下标算，不要用弹栈（栈顶弹出会变成从右往左，减法必错）。

**另一条思路**：按 `+` `-` 切分成项，每项是若干数的乘积，算出每项再按符号求和。不需要栈，代码更短。

---

## 本次暴露的写法习惯（重点）

| 问题 | 例子 | 修正 |
|------|------|------|
| 习惯"包一层" | q5 的嵌套函数 | 直接写 |
| 用补丁绕过概念 | q3 的 `append` | 用 `a[2:7:2]` |
| 忽略效率 | q2 的 `s.count` | 用 `Counter` |
| 死代码 | `return` 后的 `pass` | 删掉 |
| import 位置随意 | 文件中间 import | 统一放文件顶部 |
| 边遍历边删 | q12 的 `del` | 用标记法或遍历副本 |
