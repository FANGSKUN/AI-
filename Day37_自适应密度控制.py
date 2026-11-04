# Day 37：自适应密度控制（分裂 / 克隆 / 剪枝）
# Day 36 用固定 K=64（人猜）；今天 K 从很小的数开始，训练中自动增删
# 用 2D 高斯（Day 33 简化）聚焦密度控制本身；增删逻辑在 2D/3D 完全一样

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：目标（红圆盘，复用 Day 33）=================
def make_target(H=64, W=64, r=0.5):
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # i,j: (H,W)
    x = (j.float() - (W - 1) / 2) / (W / 2)   # x: (H,W) ∈ [-1,1]
    y = (i.float() - (H - 1) / 2) / (H / 2)   # y: (H,W)
    xy = torch.stack([x, y], dim=-1).reshape(H * W, 2)   # xy: (H*W,2)
    inside = (x ** 2 + y ** 2) < r ** 2                   # inside: (H,W)
    inside = inside.reshape(H * W)                        # inside: (H*W,)
    target = torch.where(inside[..., None],               # target: (H*W,3)
                         torch.tensor([1.0, 0.0, 0.0]),
                         torch.tensor([0.0, 0.0, 0.0]))
    return xy, target

xy, target = make_target()   # xy: (4096,2), target: (4096,3)
N = xy.shape[0]

# ================= 第 2 步：可学习参数（初始 K=12，故意很少）=================
mus = (torch.rand(12, 2) - 0.5).requires_grad_(True)        # mus: (K,2) 位置
log_sigma = torch.full((12,), np.log(0.3)).requires_grad_(True)  # log_sigma: (K,) 大小
alpha_logit = torch.full((12,), 2.0).requires_grad_(True)   # alpha_logit: (K,) 不透明度
colors = (torch.rand(12, 3) * 0.1).requires_grad_(True)     # colors: (K,3) 颜色

def gaussian(xy, mu, sigma):
    """xy (N,2), mu (2,), sigma 标量 → (N,) 高斯响应"""
    diff = xy - mu
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist_sq / (2 * sigma ** 2))

def render():
    """2D 高斯叠加（带 α 加权）→ (N,3)"""
    sigma = torch.exp(log_sigma)           # (K,) 还原大小
    alpha = torch.sigmoid(alpha_logit)     # (K,) 还原不透明度
    rgb = torch.sigmoid(colors)            # (K,3) 还原颜色
    out = torch.zeros(N, 3)
    for k in range(mus.shape[0]):
        g = gaussian(xy, mus[k], sigma[k])                # (N,)
        out = out + (alpha[k] * g).unsqueeze(-1) * rgb[k] # (N,3) 叠加
    return out

# ================= 第 3 步：密度控制（3 个 TODO）=================
def densify():
    global mus, log_sigma, alpha_logit, colors
    sigma = torch.exp(log_sigma)                 # (K,) 当前大小
    alpha = torch.sigmoid(alpha_logit)           # (K,) 当前不透明度
    g = mus.grad.norm(dim=-1)                    # (K,) 位置梯度模长（backward 后才有）
    g_threshold = g.mean()                       # 简化：梯度大于均值 = 「被拉扯得厉害」
    sigma_threshold = 0.2                        # 大小阈值：σ 大→分裂，σ 小→克隆
    alpha_threshold = 0.005                      # 透明度阈值：太透明→剪枝

    # ---- TODO 1：分裂（σ 大 且 梯度大 → 拆成 2 个小高斯）----
    # split_mask = (g > g_threshold) & (sigma > sigma_threshold)
    # 对满足的高斯 idx：原高斯 σ 减半；新加一个 σ 减半、位置加小扰动的复制品
    # ---- 在这里填 ----
    split_mask = (g > g_threshold) & (sigma > sigma_threshold)
    idx = torch.where(split_mask)[0]
    if idx.numel() > 0:
        log_sigma.data[idx] -= np.log(2)  # 原高斯 σ 减半（用 .data 避免 in-place 报错）
        # 新高斯：σ 减半 + 位置加小扰动（σ 量级）
        mus = torch.cat([mus, mus.detach()[idx] + torch.randn(idx.numel(), 2) * sigma[idx].unsqueeze(-1) * 0.3])
        log_sigma = torch.cat([log_sigma, log_sigma.detach()[idx]])  # 已经是减半后的
        alpha_logit = torch.cat([alpha_logit, alpha_logit.detach()[idx]])
        colors = torch.cat([colors, colors.detach()[idx]])

    # ---- TODO 2：克隆（σ 小 但 梯度大 → 复制 1 个）----
    # clone_mask = (g > g_threshold) & (sigma <= sigma_threshold)
    # 对满足的高斯 idx：直接复制，位置加一点小扰动
    # ---- 在这里填 ----
    clone_mask = (g > g_threshold) & (sigma <= sigma_threshold)
    idx = torch.where(clone_mask)[0]
    if idx.numel() > 0:
        mus = torch.cat([mus, mus.detach()[idx] + torch.randn(idx.numel(), 2) * 0.02])  # 位置微扰
        log_sigma = torch.cat([log_sigma, log_sigma.detach()[idx]])
        alpha_logit = torch.cat([alpha_logit, alpha_logit.detach()[idx]])
        colors = torch.cat([colors, colors.detach()[idx]])

    # ---- TODO 3：剪枝（α 太小 → 删掉）----
    # keep = alpha > alpha_threshold
    # 用布尔索引保留 keep 的高斯
    # ---- 在这里填 ----
    alpha = torch.sigmoid(alpha_logit)
    keep = alpha > alpha_threshold
    mus = mus.detach()[keep]
    log_sigma = log_sigma.detach()[keep]
    alpha_logit = alpha_logit.detach()[keep]
    colors = colors.detach()[keep]

    # 增删后：detach 断开旧计算图，重新声明为可学习叶子
    mus = mus.detach().requires_grad_(True)
    log_sigma = log_sigma.detach().requires_grad_(True)
    alpha_logit = alpha_logit.detach().requires_grad_(True)
    colors = colors.detach().requires_grad_(True)

# ================= 第 4 步：训练循环（含密度控制）=================
steps = 3000
every_n = 300
lr = 0.03
optimizer = torch.optim.Adam([mus, log_sigma, alpha_logit, colors], lr=lr)

K_history = [mus.shape[0]]                       # 记录高斯数变化
for step in range(steps):
    pred = render()
    loss = ((pred - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()

    if step > 0 and step % every_n == 0:
        K_before = mus.shape[0]
        densify()                                # 增删高斯
        K_after = mus.shape[0]
        K_history.append(K_after)
        optimizer = torch.optim.Adam([mus, log_sigma, alpha_logit, colors], lr=lr)  # 参数变了要重建优化器
        print(f"step {step:4d}  密度控制：K {K_before} → {K_after}  loss={loss.item():.5f}")
    elif step % 200 == 0:
        print(f"step {step:4d}  K={mus.shape[0]}  loss={loss.item():.5f}")

    optimizer.step()

print(f"\n训练结束：K 从 12 → {mus.shape[0]}（自动增长，不用人猜）")

# ================= 第 5 步：可视化 =================
with torch.no_grad():
    pred_final = render().clone()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target.reshape(64, 64, 3).numpy()); axes[0].set_title("目标：红圆盘"); axes[0].axis("off")
axes[1].imshow(torch.clamp(pred_final.reshape(64, 64, 3), 0, 1).numpy()); axes[1].set_title(f"训练后渲染（K={mus.shape[0]}，自动长成）"); axes[1].axis("off")

# 高斯中心散点（大小 = σ，颜色 = 各高斯 rgb）
axes[2].imshow(np.zeros((64, 64, 3)))
mu = mus.detach().numpy()
sig = torch.exp(log_sigma).detach().numpy()
rgb = torch.sigmoid(colors).detach().numpy()
for k in range(mus.shape[0]):
    px = (mu[k, 0] + 1) / 2 * 63
    py = (mu[k, 1] + 1) / 2 * 63
    axes[2].scatter(px, py, s=np.clip(sig[k] * 300, 3, 150), c=[rgb[k]], alpha=0.6, edgecolors="white", linewidths=0.3)
axes[2].set_xlim(0, 63); axes[2].set_ylim(63, 0); axes[2].set_title("高斯分布（点大=σ 大）"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("可视化图/Day37_自适应密度控制.png", dpi=120)
plt.show()

# 高斯数变化曲线
plt.figure(figsize=(5, 3))
plt.plot([i * every_n for i in range(len(K_history))], K_history, marker="o")
plt.xlabel("步数"); plt.ylabel("高斯数 K"); plt.title("自适应密度控制：K 自动增长")
plt.tight_layout()
plt.savefig("可视化图/Day37_K变化曲线.png", dpi=120)
plt.show()
