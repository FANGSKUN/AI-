# -*- coding: utf-8 -*-
"""Day17 自检：验证 4 个 TODO 是否填对。

运行：
    conda activate ai
    python Day17_自检.py
"""
import os
import torch
import Day17_检测分割 as mod

IMG = "测试图片/马.jpg"


def c1_检测模型加载():
    from torchvision.models.detection import FasterRCNN
    model, weights = mod.load_detector()
    assert isinstance(model, FasterRCNN), \
        f"检测模型应为 FasterRCNN，实际 {type(model).__name__}。检查 TODO1 的 weights 参数。"
    return "TODO1 ✓ 检测模型加载成功"


def c2_分割模型加载():
    from torchvision.models.segmentation import LRASPP
    model, weights = mod.load_segmenter()
    assert isinstance(model, LRASPP), \
        f"分割模型应为 LRASPP，实际 {type(model).__name__}。检查 TODO2 的 weights 参数。"
    return "TODO2 ✓ 分割模型加载成功"


def c3_检测输出解读():
    model, weights = mod.load_detector()
    boxes, labels, scores = mod.detect_and_draw(model, weights, IMG)
    assert boxes.ndim == 2 and boxes.shape[1] == 4, \
        f"boxes 形状应为 (N, 4)，实际 {tuple(boxes.shape)}。检查 TODO3 三个字段名。"
    assert labels.shape[0] == boxes.shape[0] == scores.shape[0], \
        "labels/scores 数量应与 boxes 一致。检查 TODO3 字段名。"
    assert os.path.exists("可视化图/Day17_检测.png"), "检测可视化图未生成，检查 detect_and_draw。"
    return f"TODO3 ✓ 检测输出 boxes{tuple(boxes.shape)} labels{tuple(labels.shape)} scores{tuple(scores.shape)}"


def c4_分割输出解读():
    model, weights = mod.load_segmenter()
    out = mod.segment_and_draw(model, weights, IMG)
    assert out.ndim == 4 and out.shape[1] == 21, \
        f"out 通道应为 21（COCO 类别数），实际形状 {tuple(out.shape)}。检查 TODO4 字段名。"
    assert os.path.exists("可视化图/Day17_分割.png"), "分割可视化图未生成，检查 segment_and_draw。"
    return f"TODO4 ✓ 分割输出 out 形状 {tuple(out.shape)}"


def main():
    checks = [c1_检测模型加载, c2_分割模型加载, c3_检测输出解读, c4_分割输出解读]
    print("=" * 56)
    all_ok = True
    for fn in checks:
        try:
            print("✓ " + fn())
        except Exception as e:
            all_ok = False
            print(f"✗ {fn.__name__} 失败：{e}")
    print("=" * 56)
    print("🎉 全部通过！" if all_ok else "有未通过的项，按提示改 TODO 后重跑。")


if __name__ == "__main__":
    main()
