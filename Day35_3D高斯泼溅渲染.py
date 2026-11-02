# Day 35：3D 高斯泼溅的渲染管线（投影 + 深度排序 + alpha 混合）
# Day 33/34 是平面泼溅（无深度、无遮挡）；真 3DGS 是 3D 高斯投影到屏幕，前面的遮挡后面的
# 今天只讲「渲染」，不做训练：5 个 3D 高斯直接给定，走完 投影→排序→混合 三步骤

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：场景（5 个 3D 高斯，直接给定）=================
# 相机在原点，看向 +z 方向（焦距 1）。z 越大越远。
mus_3d = torch.tensor([               # mus_3d: (5,3) 每个高斯的 3D 位置 (x,y,z)
    [ 0.0,  0.0, 2.0],                # 红，最近
    [ 0.3,  0.1, 3.0],                # 绿
    [-0.3,  0.0, 4.0],                # 蓝
    [ 0.1, -0.2, 2.5],                # 黄
    [ 0.0,  0.0, 6.0],                # 紫，最远
])
sigmas_3d = torch.tensor([0.3, 0.3, 0.3, 0.4, 0.5])   # sigmas_3d: (5,) 大小 σ
colors = torch.tensor([               # colors: (5,3) 颜色 rgb
    [1.0, 0.0, 0.0],                  # 红
    [0.0, 1.0, 0.0],                  # 绿
    [0.0, 0.0, 1.0],                  # 蓝
    [1.0, 1.0, 0.0],                  # 黄
    [0.8, 0.2, 1.0],                  # 紫
])
alphas = torch.tensor([0.9, 0.9, 0.9, 0.7, 0.9])       # alphas: (5,) 不透明度 α

# ================= 第 2 步：投影（TODO 1）=================
def project(mus_3d, sigmas_3d):
    """mus_3d (K,3), sigmas_3d (K,) → (mus_2d (K,2), sigmas_2d (K,))
    TODO：透视投影（焦距 1）：x2d = x/z，y2d = y/z，σ2d = σ/z（近大远小）
    提示：z = mus_3d[:, 2]；mus_2d = mus_3d[:, :2] / z.unsqueeze(-1)；sigmas_2d = sigmas_3d / z
    """
    # ---- 在这里填 ----
    z = mus_3d[:, 2]
    mus_2d = mus_3d[:, :2]/z.unsqueeze(-1)
    sigmas_2d = sigmas_3d/z
    return mus_2d, sigmas_2d
    # -------------------

# ================= 第 3 步：深度排序（TODO 2）=================
# 按深度 z 排序，得到「远 → 近」的索引顺序（远的先画，近的覆盖）
# TODO：z = mus_3d[:, 2]；order = torch.argsort(z, descending=True)（z 大的=远的排前面）
# ---- 在这里填 ----
z = mus_3d[:, 2]
order = torch.argsort(z, descending=True)
# -------------------

# ================= 第 4 步：alpha 混合渲染（TODO 3）=================
def gaussian(xy, mu2d, sigma2d):
    """xy (N,2), mu2d (2,), sigma2d 标量 → (N,) 高斯响应（复用 Day 33）"""
    diff = xy - mu2d
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist_sq / (2 * sigma2d ** 2))

def render_splat(xy, mus_2d, sigmas_2d, colors, alphas, order):
    """xy (N,2) 像素坐标 → (N,3) 从远到近 alpha 混合的渲染结果
    TODO：C 初始全 0（黑）；循环 order（远→近）：
      g = gaussian(xy, mus_2d[k], sigmas_2d[k])   # (N,) 这个高斯的 2D 响应
      a_eff = alphas[k] * g.unsqueeze(-1)         # (N,1) 有效不透明度
      C = C * (1 - a_eff) + colors[k] * a_eff     # over 合成：近的覆盖远的
    """
    # ---- 在这里填 ----
    C = torch.zeros(xy.shape[0], 3)  # (N,3) 背景 = 黑
    for k in order:  # 远 → 近
        g = gaussian(xy, mus_2d[k], sigmas_2d[k])  # (N,) 这个高斯的 2D 响应
        a_eff = alphas[k] * g.unsqueeze(-1)  # (N,1) 有效不透明度
        C = C * (1 - a_eff) + colors[k] * a_eff  # over 合成：近的覆盖远的
    return C
    # -------------------

# ================= 第 5 步：渲染 + 可视化 =================
H = W = 128
i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # i,j: (H,W)
x = (j.float() - (W - 1) / 2) / (W / 2) * 1.5   # x: (H,W) 屏幕范围 ±1.5
y = (i.float() - (H - 1) / 2) / (H / 2) * 1.5   # y: (H,W)
xy = torch.stack([x, y], dim=-1).reshape(H * W, 2)   # xy: (H*W,2)

mus_2d, sigmas_2d = project(mus_3d, sigmas_3d)       # 投影（TODO 1）
img = render_splat(xy, mus_2d, sigmas_2d, colors, alphas, order)  # 混合（TODO 3）

print("3D 高斯位置 (x,y,z)：")
for k in range(mus_3d.shape[0]):
    print(f"  高斯{k}：位置={mus_3d[k].tolist()}  σ={sigmas_3d[k].item()}  "
          f"颜色={colors[k].tolist()}  α={alphas[k].item()}  z={mus_3d[k,2].item()}")
print(f"\n深度排序（远→近）的索引：{order.tolist()}")

# 可视化：左上=渲染图，右下=渲染图+投影中心标记
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
img_show = torch.clamp(img.reshape(H, W, 3), 0, 1).numpy()
axes[0].imshow(img_show); axes[0].set_title("3D 高斯投影 + alpha 混合（近大远小 + 遮挡）")
axes[1].imshow(img_show)
mu2 = mus_2d.numpy()
for k in range(mus_2d.shape[0]):
    px = (mu2[k, 0] + 1.5) / 3 * (W - 1)
    py = (mu2[k, 1] + 1.5) / 3 * (H - 1)
    axes[1].scatter(px, py, s=60, c=[colors[k].tolist()], edgecolors="white", linewidths=1.5, zorder=5)
axes[1].set_title("高斯投影中心（颜色对应各自）")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("可视化图/Day35_3D高斯泼溅渲染.png", dpi=120)
plt.show()
