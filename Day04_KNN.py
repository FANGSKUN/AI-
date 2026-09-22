# -*- coding: utf-8 -*-
r"""
Day04 项目：纯 NumPy 手写 KNN 分类
====================================
数据：sklearn 自带的 digits（8x8 手写数字，10 类，1797 个样本）
注意：KNN 部分全部自己用 NumPy 实现，不许用 sklearn 的 KNN。

把标了 TODO 的地方补全，然后运行 Day04_自检.py 检查。

运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day04_自检.py
"""
import numpy as np
from sklearn.datasets import load_digits


# ============ 1. 数据（已提供，不用改）============

def load_data():
    """返回 X (n, 64), y (n,)。每张图拉平成 64 维向量。"""
    digits = load_digits()
    return digits.data.astype(np.float32), digits.target.astype(np.int64)


def split(X, y, test_ratio=0.2, seed=0):
    """随机划分训练集 / 测试集（已提供）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ============ 2. 距离矩阵（TODO，核心）============

def euclidean_distances(X_test, X_train):
    """返回形状 (n_test, n_train) 的距离矩阵。
    dist[i, j] = 第 i 个测试样本 到 第 j 个训练样本 的欧氏距离。

    提示（向量化，别写双重循环）：
        diff = X_test[:, None, :] - X_train[None, :, :]   # (n_test, n_train, d)
        dist = np.sqrt((diff ** 2).sum(axis=-1))           # (n_test, n_train)
    """
    # TODO
    diff = X_test[:, None, :] - X_train[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist
    pass


# ============ 3. KNN 分类（TODO，核心）============

def knn_predict(X_test, X_train, y_train, k=3):
    dist = euclidean_distances(X_test, X_train)
    idx = np.argsort(dist, axis=1)[:, :k]
    labels = y_train[idx]              # (n_test, k)

    preds = []                         # ← 用列表收集
    for row in labels:
        counts = np.bincount(row)
        preds.append(np.argmax(counts))
    return np.array(preds)             # ← 转成数组返回，形状 (n_test,)

# ============ 4. 准确率 ============

def accuracy(y_pred, y_true):
    """预测正确的比例，返回 0~1 的小数。"""
    return float(np.mean(y_pred == y_true))
