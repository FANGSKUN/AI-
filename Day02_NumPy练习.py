# -*- coding: utf-8 -*-
r"""
Day02 NumPy 练习
================
把下面每个函数补全，然后运行 Day02_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day02_自检.py

规则：尽量用 NumPy 的向量化写法，别用 Python 的 for 循环（除非题目允许）。
"""
import numpy as np

# ============ Part A  创建与属性 ============
def p1():
    """返回一个 shape=(3,4) 的全零数组，dtype 为 float32。"""
    a=np.zeros((3,4),dtype=np.float32)
    return a
    pass


def p2():
    """返回 0~9 共 10 个数，reshape 成 (2,5)。"""
    a = np.arange(10)
    a = a.reshape(2,5)
    return a
    pass


def p3(a):
    """返回 a 的属性元组：(shape, ndim, dtype, size)。"""
    b = np.array((a.shape,a.ndim,a.dtype,a.size),dtype=object)
    return tuple(b)
    pass


# ============ Part B  索引与切片 ============

def p4(a):
    """a 是 (4,5)，返回第 2 行（索引 1）。"""
    return a[1,...]
    pass


def p5(a):
    """a 是 (4,5)，返回所有行的第 1 列（索引 0），结果是一维数组。"""
    b = np.array(a[...,0])
    return b
    pass


def p6(a):
    """a 是 (4,5)，返回右下角 2x2 子矩阵。"""
    b = a[2:,3:]
    return b
    pass


def p7(a):
    """a 是任意数组，返回所有大于 10 的元素（一维）。"""
    b = a[a>10]
    return b
    pass


# ============ Part C  广播 ============

def p8():
    """令 a = np.arange(3).reshape(3,1)，b = np.arange(4).reshape(1,4)，
    返回 a + b，结果形状应为 (3,4)。"""
    a = np.arange(3).reshape(3,1)
    b = np.arange(4).reshape(1,4)
    return a+b
    pass


def p9(a):
    """a 是 (5,3)，返回"每列减去该列均值"（用广播，不要循环）。"""
    b = a.sum(axis=0)
    return a-b/5
    pass


# ============ Part D  axis ============

def p10(a):
    """a 是 (3,4)，返回元组 (a.sum(axis=0), a.sum(axis=1))。"""
    return (a.sum(axis=0), a.sum(axis=1))
    pass


def p11(a):
    """a 是 (3,4)，返回每行的均值（一维，长度 3）。"""
    b = a.sum(axis=1)/4
    b.shape=(3,)
    return b
    pass


# ============ Part E  综合 ============

def p12(x):
    """给定一维数组 x，返回标准化结果 (x - x.mean()) / x.std()。"""
    return (x - x.mean()) / x.std()
    pass


def p13(a):
    """a 是 (n, m) 矩阵，返回"每行减去该行均值"（行标准化）。"""
    b = a.sum(axis=1)/len(a[0])
    b.shape=(b.size,1)
    return a-b
    pass
