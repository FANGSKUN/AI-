# -*- coding: utf-8 -*-
r"""
Day04 手写数字识别 · 交互测试
================================
用法：
    conda activate ai
    cd "C:\Users\29322\Desktop\AI自学"
    python Day04_测试自己的数字.py

在窗口里用鼠标写一个数字（0~9），点「预测」，看你自己写的 KNN 认不认得。

原理：
  1. 你画的图（墨迹）先裁剪到数字的外框、补成正方形、缩放到 8x8
  2. 灰度映射到 0~16（和 sklearn digits 数据集一致）
  3. 喂给你写的 knn_predict
"""
import tkinter as tk

import numpy as np
from PIL import Image

from Day04_KNN import load_data, split, knn_predict

# ---------- 准备数据，训练 KNN ----------
X, y = load_data()
X_train, y_train, X_test, y_test = split(X, y)
acc = np.mean(knn_predict(X_test, X_train, y_train, 5) == y_test)
print(f"训练样本 {len(X_train)} 个，测试集准确率 {acc:.3f}")

# ---------- 画布参数 ----------
SIZE = 280
R = 12                      # 笔刷半径
img = np.zeros((SIZE, SIZE), dtype=np.float32)

yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
BRUSH = ((xx ** 2 + yy ** 2) <= R * R).astype(np.float32)


def stamp(x, y):
    """把笔刷盖到 (x, y) 处（更新 numpy 数组）。"""
    y0, y1 = max(0, y - R), min(SIZE, y + R + 1)
    x0, x1 = max(0, x - R), min(SIZE, x + R + 1)
    by0, by1 = y0 - (y - R), y1 - (y - R)
    bx0, bx1 = x0 - (x - R), x1 - (x - R)
    region = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = np.maximum(region, BRUSH[by0:by1, bx0:bx1])


_last = [None, None]


def paint(event):
    x, y = event.x, event.y
    if _last[0] is None:
        canvas.create_oval(x - R, y - R, x + R, y + R, fill="black", outline="black")
        stamp(x, y)
    else:
        canvas.create_line(_last[0], _last[1], x, y, fill="black",
                           width=2 * R, capstyle="round")
        n = int(max(abs(x - _last[0]), abs(y - _last[1]))) + 1
        for t in np.linspace(0, 1, n):
            stamp(int(_last[0] + (x - _last[0]) * t),
                  int(_last[1] + (y - _last[1]) * t))
    _last[0], _last[1] = x, y


def reset_last(event):
    _last[0] = _last[1] = None


# ---------- 预处理：画布 -> 8x8 ----------
def to_digit8x8(arr):
    ys, xs = np.where(arr > 0.1)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = arr[y0:y1, x0:x1]
    h, w = crop.shape
    s = max(h, w)
    pad = np.zeros((s, s), dtype=np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    pad[oy:oy + h, ox:ox + w] = crop
    im = Image.fromarray((pad * 255).astype(np.uint8)).resize((8, 8), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0 * 16.0


ASCII = " .:-=+*#%@"


def ascii_art(small):
    return "\n".join(
        "".join(ASCII[min(9, int(v / 16 * 9.99))] for v in row) for row in small
    )


# ---------- 按钮动作 ----------
def predict():
    small = to_digit8x8(img)
    if small is None:
        result_label.config(text="请先写一个数字")
        small_label.config(text="")
        return
    pred = knn_predict(small.reshape(1, -1), X_train, y_train, k=3)
    result_label.config(text=f"预测：{int(pred[0])}")
    small_label.config(text=ascii_art(small))


def clear():
    canvas.delete("all")
    img[:] = 0
    result_label.config(text="预测：?")
    small_label.config(text="")


# ---------- 界面 ----------
root = tk.Tk()
root.title("手写数字识别 · 纯 NumPy KNN")

canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="white")
canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=10)
canvas.bind("<Button-1>", paint)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset_last)

tk.Button(root, text="预测", command=predict, width=14, font=("Arial", 12)).grid(
    row=0, column=1, pady=6)
tk.Button(root, text="清除", command=clear, width=14, font=("Arial", 12)).grid(
    row=1, column=1, pady=6)

result_label = tk.Label(root, text="预测：?", font=("Arial", 22))
result_label.grid(row=2, column=1, pady=10)

tk.Label(root, text="模型看到的 8x8：", font=("Arial", 10)).grid(row=3, column=1)
small_label = tk.Label(root, text="", font=("Consolas", 11), justify="left")
small_label.grid(row=4, column=1, pady=6)

root.mainloop()
