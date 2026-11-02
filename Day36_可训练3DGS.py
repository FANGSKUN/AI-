# Day 36：可训练的真 3DGS（完整闭环）
# = 可微优化（Day 33/34 训练高斯）+ 渲染管线（Day 35 投影+排序+混合）
# 目标：训练 K 个 3D 高斯，投影渲染出一张 2D 图，拟合目标红圆盘

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：目标图（红圆盘，复用 Day 33）=================
def make_target(H=64, W=64, r=0.5):
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # i,j: (H,W)
    x = (j.float() - (W - 1) / 2) / (W / 2)   # x: (H,W) ∈ [-1,1]
    y = (i.float() - (H - 1) / 2) / (H / 2)   # y: (H,W) ∈ [-1,1]
    xy = torch.stack([x, y], dim=-1).reshape(H * W, 2)   # xy: (H*W,2)
    inside = (x ** 2 + y ** 2) < r ** 2                   # inside: (H,W) 圆内
    inside = inside.reshape(H * W)                        # inside: (H*W,)
    target = torch.where(inside[..., None],               # target: (H*W,3)
                         torch.tensor([1.0, 0.0, 0.0]),
                         torch.tensor([0.0, 0.0, 0.0]))
    return xy, target

xy, target = make_target()   # xy: (4096,2), target: (4096,3) 红圆盘
N = xy.shape[0]

# ================= 第 2 步：可学习参数（TODO 1）=================
K = 64
# TODO 1：初始化 K 个 3D 高斯的 4 组可学习参数（都 requires_grad）
#   位置 mus_3d (K,3)：x,y ∈ [-0.5,0.5]、z ∈ [2,4]（相机前方，z 越大越远）
#   大小 log_sigma (K,)：对数参数化，初始 log(0.3)（exp 还原恒正）
#   不透明度 alpha_logit (K,)：sigmoid 参数化，初始 2.0（sigmoid(2.0)≈0.88）
#   颜色 colors (K,3)：随机 [0,1]，用 sigmoid 锁 [0,1]
# ---- 在这里填 ----
mus_3d = torch.rand(K, 3).requires_grad_(True)      # 位置 (K,3)
mus_3d.data[:, 0] = mus_3d.data[:, 0] * 1.0 - 0.5   # x ∈ [-0.5, 0.5]
mus_3d.data[:, 1] = mus_3d.data[:, 1] * 1.0 - 0.5   # y ∈ [-0.5, 0.5]
mus_3d.data[:, 2] = mus_3d.data[:, 2] * 2.0 + 2.0   # z ∈ [2, 4]（相机前方）

log_sigma   = torch.full((K,), np.log(0.3)).requires_grad_(True)  # 大小（log 参数化）
alpha_logit = torch.full((K,), 2.0).requires_grad_(True)          # 不透明度（sigmoid 参数化）
colors      = torch.rand(K, 3).requires_grad_(True)               # 颜色（sigmoid 锁 [0,1]）
# -------------------

# ================= 第 3 步：渲染函数（TODO 2）=================
def gaussian(xy, mu2d, sigma2d):
    """xy (N,2), mu2d (2,), sigma2d 标量 → (N,) 高斯响应（复用 Day 33）"""
    diff = xy - mu2d
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist_sq / (2 * sigma2d ** 2))

def render_current():
    """用当前参数渲染 → (N,3)
    TODO 2：还原参数 + 投影 + 深度排序 + alpha 混合（Day 35 管线 + 参数化）
      ① 还原：sigma = exp(log_sigma)；alpha = sigmoid(alpha_logit)；rgb = sigmoid(colors)
      ② 投影：z = mus_3d[:,2].clamp(min=0.1)；mu2d = mus_3d[:,:2]/z；sigma2d = sigma/z
      ③ 排序：order = argsort(z, descending=True)
      ④ 混合：C 全 0；从远到近 C = C*(1-a) + rgb[k]*a，a = alpha[k]*g
    """
    # ---- 在这里填 ----
    sigma = torch.exp(log_sigma)  # (K,) 还原大小（恒正）
    alpha = torch.sigmoid(alpha_logit)  # (K,) 还原不透明度 [0,1]
    rgb = torch.sigmoid(colors)  # (K,3) 还原颜色 [0,1]
    z = mus_3d[:, 2].clamp(min=0.1)  # (K,) 深度，clamp 防负数
    mu2d = mus_3d[:, :2] / z.unsqueeze(-1)  # (K,2) 投影 x/z, y/z
    sigma2d = sigma / z  # (K,) 近大远小
    order = torch.argsort(z, descending=True)  # 远→近
    C = torch.zeros(N, 3)  # 背景黑
    for k in order:  # 从远到近
        g = gaussian(xy, mu2d[k], sigma2d[k])  # (N,) 光斑
        a = alpha[k] * g.unsqueeze(-1)  # (N,1) 有效不透明度
        C = C * (1 - a) + rgb[k] * a  # over 合成
    return C
    # -------------------

# ================= 第 4 步：训练循环（TODO 3）=================
# TODO 3：Adam 优化器 + 训练循环（Day 33/34 同款）
#   优化器：torch.optim.Adam([mus_3d, log_sigma, alpha_logit, colors], lr=0.05)
#   循环 2000 步：pred = render_current() → loss = MSE(pred, target)
#     → optimizer.zero_grad() → loss.backward() → optimizer.step()
#   每 200 步打印 loss
# ---- 在这里填 ----
optimizer = torch.optim.Adam([mus_3d, log_sigma, alpha_logit, colors], lr=0.05)
for step in range(2000):
    pred = render_current()
    loss = ((pred - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss = {loss.item():.6f}")
# -------------------

# ================= 第 5 步：可视化 =================
with torch.no_grad():
    pred_final = render_current().clone()

def show_img(ax, img, title):
    ax.imshow(torch.clamp(img.reshape(64, 64, 3), 0, 1).numpy())
    ax.set_title(title); ax.axis("off")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
show_img(axes[0], target, "目标：红圆盘")
show_img(axes[1], pred_final, f"3D 高斯训练后渲染（K={K}）")

# 右图：高斯投影中心散点（颜色=各自 rgb，点大小=投影后的 σ）
mu2d = (mus_3d[:, :2] / mus_3d[:, 2:3].clamp(min=0.1)).detach().numpy()  # (K,2)
rgb = torch.sigmoid(colors).detach().numpy()                               # (K,3)
sigma2d = (torch.exp(log_sigma) / mus_3d[:, 2].clamp(min=0.1)).detach().numpy()  # (K,)
axes[2].imshow(np.zeros((64, 64, 3)))
for k in range(K):
    px = (mu2d[k, 0] + 1) / 2 * 63
    py = (mu2d[k, 1] + 1) / 2 * 63
    axes[2].scatter(px, py, s=np.clip(sigma2d[k] * 400, 5, 200), c=[rgb[k]], alpha=0.7, edgecolors="white", linewidths=0.4)
axes[2].set_xlim(0, 63); axes[2].set_ylim(63, 0); axes[2].set_title("3D 高斯投影中心（点大=近，小=远）"); axes[2].axis("off")

plt.tight_layout()
plt.savefig("可视化图/Day36_可训练3DGS.png", dpi=120)
plt.show()
