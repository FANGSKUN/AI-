# -*- coding: utf-8 -*-
r"""
Day17 项目：目标检测 / 语义分割（入门）
=====================================
从「这张图是什么」（分类）升级到：
  1. 目标检测：物体在哪 + 是什么 → 输出一堆「边界框」
  2. 语义分割：每个像素是什么 → 输出一张「标签图」

三个任务的区别（今天的核心概念）：

  分类（Day15/16） 输入一张图 → 输出 1 个类别        「这是猫」
  目标检测         输入一张图 → 输出 N 个框+类别      「这有只猫，框在(x,y,w,h)」
  语义分割         输入一张图 → 输出每个像素的类别     「这些像素是猫」

今天**不从零训练**（检测/分割从零训练很重），而是用 torchvision 的
**预训练模型**（已在 COCO 数据集上训练好），在你的照片上跑，直观体验。

预训练 = 别人用超大数据集训练好的权重直接拿来用 —— 这正是 Day15 说的
「治本」手段（模型见够了真实世界的画风）。

运行：
    conda activate ai
    python Day17_自检.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# VOC 21 类经典颜色表（语义分割可视化用，每类一个固定颜色）
VOC_COLORS = np.array([
    [0, 0, 0],        # 0  背景
    [128, 0, 0],      # 1  飞机
    [0, 128, 0],      # 2  自行车
    [128, 128, 0],    # 3  鸟
    [0, 0, 128],      # 4  船
    [128, 0, 128],    # 5  瓶子
    [0, 128, 128],    # 6  公交车
    [128, 128, 128],  # 7  汽车
    [64, 0, 0],       # 8  猫
    [192, 0, 0],      # 9  椅子
    [64, 128, 0],     # 10 牛
    [192, 128, 0],    # 11 餐桌
    [64, 0, 128],     # 12 狗
    [192, 0, 128],    # 13 马
    [64, 128, 128],   # 14 摩托车
    [192, 128, 128],  # 15 人
    [0, 64, 0],       # 16 盆栽
    [128, 64, 0],     # 17 羊
    [0, 192, 0],      # 18 沙发
    [128, 192, 0],    # 19 火车
    [0, 64, 128],     # 20 电视
], dtype=np.uint8)

# COCO 检测 80 类常用项的中文映射（其余显示英文）
DET_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马",
    "sheep": "羊", "cow": "牛", "elephant": "大象", "bear": "熊",
    "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "suitcase": "行李箱", "skis": "滑雪板",
    "snowboard": "滑雪板", "sports ball": "球", "kite": "风筝",
    "skateboard": "滑板", "surfboard": "冲浪板", "tennis racket": "网球拍",
    "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉",
    "apple": "苹果", "sandwich": "三明治", "orange": "橙子", "broccoli": "西兰花",
    "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈",
    "cake": "蛋糕", "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
    "laptop": "笔记本", "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘",
    "cell phone": "手机", "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机",
    "sink": "水槽", "refrigerator": "冰箱", "book": "书", "clock": "时钟",
    "vase": "花瓶", "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机",
    "toothbrush": "牙刷",
}


# ============ 1. 加载预训练模型（4 个 TODO） ============

def load_detector():
    """目标检测模型：Faster R-CNN（MobileNet V3 骨干，轻量）。"""
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    )
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    # TODO 1：加载检测模型。weights 参数该填什么？（填上面的变量名）
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.eval().to(device)
    return model, weights


def load_segmenter():
    """语义分割模型：LRASPP（MobileNet V3 骨干，轻量）。"""
    from torchvision.models.segmentation import (
        lraspp_mobilenet_v3_large,
        LRASPP_MobileNet_V3_Large_Weights,
    )
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    # TODO 2：加载分割模型。weights 参数该填什么？
    model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval().to(device)
    return model, weights


# ============ 2. 检测：解读输出 + 画框 ============

def detect_and_draw(model, weights, image_path, score_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    transform = weights.transforms()
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0]

    # TODO 3：检测输出是一个字典，三个关键字段——
    boxes = pred["boxes"]    # 每个物体的边界框，形状 (N, 4)
    labels = pred["labels"]   # 每个框的类别编号，形状 (N,)
    scores = pred["scores"]   # 每个框的置信度，形状 (N,)

    categories = weights.meta["categories"]  # COCO 80 类英文名
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        name_en = categories[label]
        name = DET_ZH.get(name_en, name_en)
        ax.text(x1, y1 - 5, f"{name} {score:.2f}", color="red", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7))
    ax.axis("off")
    ax.set_title(f"目标检测：{image_path}")
    plt.tight_layout()
    plt.savefig("可视化图/Day17_检测.png", dpi=120, bbox_inches="tight")
    plt.close()
    return boxes, labels, scores


# ============ 3. 分割：解读输出 + 显示掩码 ============

def segment_and_draw(model, weights, image_path):
    img = Image.open(image_path).convert("RGB")
    transform = weights.transforms()
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)

    # TODO 4：分割输出也是字典，关键字段 "out"，形状 (1, 21, H, W)
    out = output["out"]
    # 对每个像素在 21 个类别里取 argmax → 得到 (H, W) 的类别图
    # 提示：out[0] 形状是 (21, H, W)，要沿「通道维」取 argmax，dim = ?
    seg_map = out[0].argmax(dim=0)

    seg_np = seg_map.cpu().numpy().astype(np.uint8)

    # 用 VOC 颜色表给每个类别染色（马=13 → 紫红色）
    color_mask = VOC_COLORS[seg_np]                      # (H', W', 3) 彩色图
    color_mask = np.array(Image.fromarray(color_mask).resize(img.size, Image.NEAREST))

    # 前景（非背景）区域半透明染上类别色，背景保持原样
    fg = (seg_np > 0)
    fg = np.array(Image.fromarray((fg * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)) > 127
    img_np = np.array(img)
    overlay = img_np.copy()
    overlay[fg] = (overlay[fg] * 0.3 + color_mask[fg] * 0.7).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("原图")
    axes[0].axis("off")
    axes[1].imshow(color_mask)
    axes[1].set_title("分割掩码（VOC 颜色表）")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("叠加显示（前景染色）")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("可视化图/Day17_分割.png", dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ============ 4. 主程序 ============

def main():
    detector, d_weights = load_detector()
    segmenter, s_weights = load_segmenter()

    img = "测试图片/马.jpg"
    print(f"=== 目标检测：{img} ===")
    boxes, labels, scores = detect_and_draw(detector, d_weights, img)
    print(f"检测到 {len(boxes)} 个候选框（含低置信度，图里只画 > 0.5 的）")

    print(f"\n=== 语义分割：{img} ===")
    segment_and_draw(segmenter, s_weights, img)
    print("分割完成，可视化图已保存到 可视化图/")


if __name__ == "__main__":
    main()
