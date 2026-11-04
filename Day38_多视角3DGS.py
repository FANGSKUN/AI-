# Day 38：多视角 3DGS——解决单视角 z 冗余（尺度模糊）
# 关键概念修正：锁 z 靠「相机平移产生的基线/视差」，不靠「绕原点旋转」
#   单相机（光心不动）：x2d = x/z 只有比值 → z 冗余（尺度模糊，scale ambiguity）
#   双相机（基线 b）：  x2d₁ - x2d₂ = b/z → z = b/视差（= Day 23 三角化）
# 纯旋转为什么不行：旋转是线性变换 R(λp)=λR(p)，整体缩放 λ 旋转后投影不变

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 场景：一个红高斯 A（真实位置已知）=================
mus_A = torch.tensor([[0.3, 0.2, 2.5]])     # mus_A: (1,3) 真实 3D 位置
sigmas_A = torch.tensor([0.3])              # sigmas_A: (1,) 真实大小
colors_A = torch.tensor([[1.0, 0.0, 0.0]])  # colors_A: (1,3) 红色
alpha = torch.tensor([0.9])                 # alpha: (1,) 不透明度

# ================= TODO 1：相机平移基线投影 =================
def project_view(mus_3d, sigmas_3d, cam_x):
    """mus_3d (K,3), sigmas_3d (K,), cam_x 相机在 x 轴的位置（基线）→ (mu2d (K,2), sigma2d (K,))
    TODO：相机在 (cam_x, 0, 0)，看向 +z 方向
      x2d = (x - cam_x) / z，y2d = y / z，σ2d = σ / z（近大远小）
    提示：z 要 clamp(min=0.1) 防负数
    """
    # ---- 在这里填 ----
    x, y, z = mus_3d[:, 0], mus_3d[:, 1], mus_3d[:, 2]
    z = z.clamp(min=0.1)  # 防负数
    mu2d = torch.stack([(x - cam_x) / z, y / z], dim=-1)  # 关键：(x - cam_x)/z 而非 x/z
    sigma2d = sigmas_3d / z  # 近大远小
    return mu2d, sigma2d

    # -------------------

# ================= TODO 2：单相机渲染（投影 + 排序 + 混合）=================
def gaussian(xy, mu2d, sigma2d):
    diff = xy - mu2d
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist_sq / (2 * sigma2d ** 2))

def render_view(mus_3d, sigmas_3d, colors, alpha, cam_x, xy):
    """从相机位置 cam_x 渲染 → (N,3)
    TODO：投影 → 深度排序 → alpha 混合（复用 Day 35/36 管线）
      1. mu2d, sigma2d = project_view(...)
      2. order = argsort(深度 z, descending=True)
      3. C 全 0；从远到近 C = C*(1-a) + color*a
    """
    # ---- 在这里填 ----
    mu2d, sigma2d = project_view(mus_3d, sigmas_3d, cam_x)
    z = mus_3d[:, 2].clamp(min=0.1)
    order = torch.argsort(z, descending=True)  # 远→近
    C = torch.zeros(xy.shape[0], 3)
    for k in order:
        g = gaussian(xy, mu2d[k], sigma2d[k])
        a = alpha[k] * g.unsqueeze(-1)
        C = C * (1 - a) + colors[k] * a
    return C

    # -------------------

# ================= 屏幕坐标 =================
H = W = 64
i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
x = (j.float() - (W - 1) / 2) / (W / 2) * 1.5   # x: (H,W) 屏幕 ±1.5
y = (i.float() - (H - 1) / 2) / (H / 2) * 1.5
xy = torch.stack([x, y], dim=-1).reshape(H * W, 2)   # xy: (H*W,2)

# ================= Part 1：构造法展示 z 冗余（读 + 跑，不用填）=================
print("=" * 56)
print("Part 1：构造「影子高斯」B = A 整体放大 2 倍")
scale = 2.0
mus_B = mus_A * scale        # (0.6, 0.4, 5.0)
sigmas_B = sigmas_A * scale  # 0.6
# A 和 B 是同一个方向、不同深度的两个点：单相机投影一模一样（z 冗余）

rA_0 = render_view(mus_A, sigmas_A, colors_A, alpha, 0.0, xy)
rB_0 = render_view(mus_B, sigmas_B, colors_A, alpha, 0.0, xy)
print(f"单相机 cam=0.0：A vs B 渲染差 = {(rA_0 - rB_0).abs().mean().item():.6f}  ← 分不清（z 冗余）")

rA_b = render_view(mus_A, sigmas_A, colors_A, alpha, 0.8, xy)
rB_b = render_view(mus_B, sigmas_B, colors_A, alpha, 0.8, xy)
print(f"双相机 cam=0.8：A vs B 渲染差 = {(rA_b - rB_b).abs().mean().item():.6f}  ← 能区分（基线视差锁 z）")
print("=" * 56)

# ================= 训练目标图（用真实 A 从 3 个相机位置渲染）=================
cams = [0.0, 0.8, -0.8]     # 3 个相机位置（基线）
targets = [render_view(mus_A, sigmas_A, colors_A, alpha, c, xy) for c in cams]  # targets: [3×(N,3)]

# ================= TODO 3：训练（单相机 vs 多相机）=================
def train(cam_list, steps=2000, lr=0.05, verbose=True):
    """用 cam_list 里的相机训练 → 返回训练后的 (mus_3d, sigma)
    TODO：初始化 1 个高斯（随机位置 + logσ 初始 log(0.5)），拟合 cam_list 所有相机目标
      损失 = 所有相机渲染和目标 MSE 之和
    提示：只训练 mus_3d 和 log_sigma；颜色用 colors_A 固定
    """
    # ---- 在这里填 ----
    mus_3d = torch.rand(1, 3).requires_grad_(True)
    mus_3d.data[0, 0] = mus_3d.data[0, 0] * 1.0 - 0.5  # x
    mus_3d.data[0, 1] = mus_3d.data[0, 1] * 1.0 - 0.5  # y
    mus_3d.data[0, 2] = mus_3d.data[0, 2] * 2.0 + 2.0  # z ∈ [2,4]
    log_sigma = torch.full((1,), np.log(0.5)).requires_grad_(True)  # σ0=0.5≠真实0.3

    idx_list = [cams.index(c) for c in cam_list]  # 相机位置 → target 索引

    optimizer = torch.optim.Adam([mus_3d, log_sigma], lr=lr)
    for step in range(steps):
        loss = torch.tensor(0.0)
        for i, c in enumerate(cam_list):
            pred = render_view(mus_3d, torch.exp(log_sigma), colors_A, alpha, c, xy)
            loss = loss + ((pred - targets[idx_list[i]]) ** 2).mean()
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()

    # -------------------
    if verbose:
        print(f"训练后位置：{mus_3d.detach().numpy()}")
    return mus_3d.detach(), torch.exp(log_sigma).detach()

# ================= 训练对比 =================
print("\n真实位置：", mus_A.numpy()[0], " σ=0.3")
print("\n【单相机训练】只用 cam=0（光心不动）→ z 会漂")
mus_s, sig_s = train([0.0])
print(f"z 误差：{abs(mus_s[0, 2].item() - 2.5):.4f}")

print("\n【多相机训练】用 3 个相机（有基线）→ z 被锁定")
mus_m, sig_m = train([0.0, 0.8, -0.8])
print(f"z 误差：{abs(mus_m[0, 2].item() - 2.5):.4f}")
print(f"位置误差：{(mus_m - mus_A).norm().item():.4f}")

# ================= 可视化 =================
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
with torch.no_grad():
    axes[0].imshow(torch.clamp(rA_0.reshape(H, W, 3), 0, 1).numpy())
    axes[0].set_title("A·单相机 cam=0"); axes[0].axis("off")
    axes[1].imshow(torch.clamp(rB_0.reshape(H, W, 3), 0, 1).numpy())
    axes[1].set_title("B(=A×2)·单相机（和 A 一样）"); axes[1].axis("off")
    axes[2].imshow(torch.clamp(rA_b.reshape(H, W, 3), 0, 1).numpy())
    axes[2].set_title("A·双相机 cam=0.8"); axes[2].axis("off")
    axes[3].imshow(torch.clamp(rB_b.reshape(H, W, 3), 0, 1).numpy())
    axes[3].set_title("B·双相机（和 A 不同 → 锁 z）"); axes[3].axis("off")
plt.tight_layout()
plt.savefig("可视化图/Day38_多视角3DGS.png", dpi=120)
plt.show()
