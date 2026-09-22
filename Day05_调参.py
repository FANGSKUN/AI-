# -*- coding: utf-8 -*-
r"""
Day05 任务1：KNN 调参 —— 不同 k 的准确率
==========================================
运行：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day05_调参.py

观察：k 从 1 变大时，准确率怎么变？为什么？
"""
import matplotlib.pyplot as plt
import numpy as np

from Day04_KNN import load_data, split, knn_predict

# 中文显示（避免方框）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

X, y = load_data()
X_train, y_train, X_test, y_test = split(X, y)

ks = [1, 3, 5, 7, 9, 11, 15, 20, 30]
accs = []

for k in ks:
    pred = knn_predict(X_test, X_train, y_train, k)
    acc = np.mean(pred == y_test)
    accs.append(acc)
    print(f"k = {k:2d}   准确率 = {acc:.4f}")

best_k = ks[int(np.argmax(accs))]
print(f"\n最好的 k = {best_k}，准确率 = {max(accs):.4f}")

plt.figure(figsize=(7, 4))
plt.plot(ks, accs, marker="o")
plt.xlabel("k（邻居个数）")
plt.ylabel("测试集准确率")
plt.title("KNN 准确率 随 k 的变化")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
