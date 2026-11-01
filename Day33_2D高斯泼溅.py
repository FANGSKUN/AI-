# Day 33：2D 高斯泼溅（3DGS 入门）—— 显式表示 vs 隐式表示
# 用 K 个 2D 高斯「泼」出红圆盘，对比 Day 30 用 MLP「背」出红圆盘
# 核心 4 连：显式表示 / 高斯三要素（位置μ·大小σ·颜色c）/ 泼溅叠加 / 可微优化

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：目标图（红圆盘）+ 像素坐标 =================
def make_grid(H=64, W=64):
    """生成像素坐标 xy (H*W,2) 和目标图 target (H*W,3)：圆内红、圆外黑"""
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # i,j: (H,W)
    x = (j.float() - (W - 1) / 2) / (W / 2)   # x: (H,W) ∈ [-1,1]
    y = (i.float() - (H - 1) / 2) / (H / 2)   # y: (H,W) ∈ [-1,1]
    xy = torch.stack([x, y], dim=-1).reshape(H * W, 2)  # xy: (H*W,2) 每个像素的 (x,y)
    r = torch.sqrt(x**2 + y**2).reshape(H * W)          # r: (H*W,) 到圆心距离
    inside = r < 0.4                                    # inside: (H*W,) 圆内 True
    target = torch.where(inside[..., None],             # target: (H*W,3) 圆内红、圆外黑
                         torch.tensor([1.0, 0.0, 0.0]),
                         torch.tensor([0.0, 0.0, 0.0]))
    return xy, target

# ================= 第 2 步：高斯参数（可学习 = 显式表示）=================
# 对比 Day 30：那里场景存在 MLP 权重里（黑盒、隐式）；这里场景就是这 32 个高斯（看得见、显式）
K = 32                                                              # 32 个高斯
means = (torch.rand(K, 2) - 0.5).requires_grad_(True)               # means: (K,2) 位置 μ，散布 [-0.5,0.5]
log_sigmas = torch.full((K,), np.log(0.1)).requires_grad_(True)     # log_sigmas: (K,) 大小 σ（取对数保证恒正）
colors = (torch.rand(K, 3) * 0.1).requires_grad_(True)              # colors: (K,3) 颜色（含亮度）

# ================= 第 3 步：单个高斯响应（TODO 1）=================
def gaussian(xy, mu, sigma):
    """xy (N,2) 像素坐标, mu (2,) 高斯中心, sigma 标量(大小) → (N,) 高斯响应
    TODO：g = exp(-||xy - mu||² / (2σ²))
    提示：diff = xy - mu（广播相减）→ 平方和 sum(dim=-1) → exp
    """
    # ---- 在这里填 ----
    diff = xy - mu
    dist_sq = (diff ** 2).sum(dim=-1)  # (N,) 到中心的距离平方
    return torch.exp(-dist_sq / (2 * sigma ** 2))
    # -------------------

# ================= 第 4 步：泼溅叠加（TODO 2）=================
def render(xy, means, sigmas, colors):
    """xy (N,2), means (K,2), sigmas (K,), colors (K,3) → (N,3) 泼溅渲染
    TODO：对每个高斯 k，算响应 g (N,)，乘颜色 colors[k] (3,)，累加到 out
    提示：out 初始 torch.zeros(N,3)；循环 k：g = gaussian(...)；out += g.unsqueeze(-1) * colors[k]
    """
    # ---- 在这里填 ----
    N = xy.shape[0]
    out = torch.zeros(N,3)
    for k in range(means.shape[0]):
        g = gaussian(xy, means[k], sigmas[k])
        out += g.unsqueeze(-1) * colors[k]
    return out
    # -------------------

# ================= 第 5 步：训练（TODO 3）=================
xy, target = make_grid()

params = [means, log_sigmas, colors]
opt = torch.optim.Adam(params, lr=1e-2)

def render_current():
    sigmas = torch.exp(log_sigmas)                  # sigmas: (K,) 从对数还原，保证正
    return render(xy, means, sigmas, colors)        # (N,3)

pred_init = render_current().detach()               # 训练前渲染（对比用）

print("开始训练（2D 高斯泼溅）...")
for step in range(3000):
    # ---- 在这里填（TODO 3）：渲染 → loss → 反传 → 更新 ----
    pred = render_current()  # 渲染
    loss = F.mse_loss(pred, target)  # 算误差
    opt.zero_grad()  # 清梯度
    loss.backward()  # 反传
    opt.step()  # 更新高斯参数
    # -------------------
    if step % 300 == 0:
        print(f"  step {step:4d}  loss {loss.item():.6f}")

pred_final = render_current().detach()
print(f"最终 loss = {F.mse_loss(pred_final, target).item():.6f}")

# ================= 第 6 步：可视化 =================
def to_img(v, H=64, W=64):
    return torch.clamp(v.reshape(H, W, 3), 0, 1).numpy()   # (H,W,3) 数值钳到 [0,1]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(to_img(target)); axes[0, 0].set_title("目标：红圆盘")
axes[0, 1].imshow(to_img(pred_init)); axes[0, 1].set_title("训练前（随机高斯泼溅）")
axes[1, 0].imshow(to_img(pred_final)); axes[1, 0].set_title("训练后（高斯拟合）")
# 右下：训练后渲染 + 高斯中心散点（显式表示的直观证据：能看见每个高斯在哪）
axes[1, 1].imshow(to_img(pred_final))
H = W = 64                                                 # 分辨率（make_grid 默认值）
mu = means.detach().numpy()                                 # mu: (K,2)
px = (mu[:, 0] + 1) / 2 * (W - 1)                           # x∈[-1,1] → 像素列
py = (mu[:, 1] + 1) / 2 * (H - 1)                           # y∈[-1,1] → 像素行
axes[1, 1].scatter(px, py, c="white", s=10, edgecolors="k", linewidths=0.5)
axes[1, 1].set_title(f"训练后 + 高斯中心（{K} 个）")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("可视化图/Day33_2D高斯泼溅.png", dpi=120)
plt.show()
