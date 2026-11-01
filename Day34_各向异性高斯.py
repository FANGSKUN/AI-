# Day 34：各向异性高斯（椭圆高斯）——协方差矩阵入门
# Day 33 的圆高斯（各向同性）是「胖」的，贴不住细长形状；今天让高斯能「压扁」，贴住目标
# 核心：各向同性(圆/球) vs 各向异性(椭圆/椭球)，椭圆高斯 = 两轴(σx,σy) + 旋转(θ)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：目标图（细长椭圆）+ 像素坐标 =================
def make_target(H=64, W=64, a=0.6, b=0.15):
    """生成横扁椭圆目标 target (H*W,3)：椭圆内红、外黑。a=长半轴, b=短半轴"""
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # i,j: (H,W)
    x = (j.float() - (W - 1) / 2) / (W / 2)   # x: (H,W) ∈ [-1,1]
    y = (i.float() - (H - 1) / 2) / (H / 2)   # y: (H,W) ∈ [-1,1]
    xy = torch.stack([x, y], dim=-1).reshape(H * W, 2)  # xy: (H*W,2)
    inside = ((x / a) ** 2 + (y / b) ** 2) < 1           # inside: (H,W) 椭圆方程 < 1
    inside = inside.reshape(H * W)                       # inside: (H*W,)
    target = torch.where(inside[..., None],              # target: (H*W,3) 椭圆内红、外黑
                         torch.tensor([1.0, 0.0, 0.0]),
                         torch.tensor([0.0, 0.0, 0.0]))
    return xy, target

# ================= 第 2 步：椭圆高斯参数（6 个旋钮）=================
K = 48                                                               # 48 个椭圆高斯
means = (torch.rand(K, 2) - 0.5).requires_grad_(True)               # means: (K,2) 位置
log_sigma_x = torch.full((K,), np.log(0.15)).requires_grad_(True)   # log_sigma_x: (K,) x 轴大小
log_sigma_y = torch.full((K,), np.log(0.15)).requires_grad_(True)   # log_sigma_y: (K,) y 轴大小
thetas = (torch.rand(K) * np.pi).requires_grad_(True)               # thetas: (K,) 旋转角 [0,π)
colors = (torch.rand(K, 3) * 0.1).requires_grad_(True)              # colors: (K,3) 颜色

# ================= 第 3 步：椭圆高斯响应（TODO 1）=================
def gaussian_aniso(xy, mu, sigma_x, sigma_y, theta):
    """xy (N,2), mu (2,), sigma_x/sigma_y 标量(两轴), theta 标量(旋转角) → (N,) 椭圆高斯响应
    TODO：平移 → 旋转 -theta（转到椭圆主轴坐标系）→ 两轴分别套高斯
    提示：
      dx = xy - mu                                    # (N,2) 平移到中心
      cos_t = torch.cos(theta); sin_t = torch.sin(theta)
      x_rot =  cos_t * dx[:,0] + sin_t * dx[:,1]      # (N,) 旋转后的 x'
      y_rot = -sin_t * dx[:,0] + cos_t * dx[:,1]      # (N,) 旋转后的 y'
      return torch.exp(-(x_rot**2/(2*sigma_x**2) + y_rot**2/(2*sigma_y**2)))
    """
    # ---- 在这里填 ----
    dx = xy -mu
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = cos_t * dx[:,0] + sin_t * dx[:,1]
    y_rot = -sin_t * dx[:,0] + cos_t * dx[:,1]
    return torch.exp(-(x_rot**2/(2*sigma_x**2) + y_rot**2/(2*sigma_y**2)))
    # -------------------

# ================= 第 4 步：泼溅叠加（TODO 2）=================
def render(xy, means, sigma_x, sigma_y, thetas, colors):
    """xy (N,2), means (K,2), sigma_x/sigma_y/thetas (K,), colors (K,3) → (N,3) 泼溅渲染
    TODO：循环 K 个椭圆高斯，算响应 g (N,)，乘颜色，累加（复用 Day 33 思路）
    提示：out 初始 torch.zeros(xy.shape[0],3)；循环 k：g = gaussian_aniso(...)；out += g.unsqueeze(-1) * colors[k]
    """
    # ---- 在这里填 ----
    out = torch.zeros(xy.shape[0], 3)
    for k in range(means.shape[0]):
        g = gaussian_aniso(xy, means[k], sigma_x[k], sigma_y[k], thetas[k])
        out += g.unsqueeze(-1) * colors[k]
    return out
    # -------------------

# ================= 第 5 步：训练（TODO 3）=================
xy, target = make_target()

params = [means, log_sigma_x, log_sigma_y, thetas, colors]
opt = torch.optim.Adam(params, lr=1e-2)

def render_current():
    sx = torch.exp(log_sigma_x)                       # sx: (K,) 保证正
    sy = torch.exp(log_sigma_y)                       # sy: (K,)
    return render(xy, means, sx, sy, thetas, colors)  # (N,3)

pred_init = render_current().detach()                 # 训练前渲染

print("开始训练（各向异性高斯）...")
for step in range(4000):
    # ---- 在这里填（TODO 3）：渲染 → loss → 反传 → 更新 ----
    pred_final = render_current()
    loss = F.mse_loss(pred_final, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    # -------------------
    if step % 400 == 0:
        print(f"  step {step:4d}  loss {loss.item():.6f}")

pred_final = render_current().detach()
print(f"最终 loss = {F.mse_loss(pred_final, target).item():.6f}")

# ================= 第 6 步：可视化 =================
def to_img(v, H=64, W=64):
    return torch.clamp(v.reshape(H, W, 3), 0, 1).numpy()

H = W = 64
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(to_img(target)); axes[0, 0].set_title("目标：细长椭圆")
axes[0, 1].imshow(to_img(pred_init)); axes[0, 1].set_title("训练前（随机椭圆高斯）")
axes[1, 0].imshow(to_img(pred_final)); axes[1, 0].set_title("训练后（高斯压扁贴住）")
# 右下：训练后渲染 + 每个高斯的椭圆形状（各向异性的直观证据）
axes[1, 1].imshow(to_img(pred_final))
mu = means.detach().numpy()                             # mu: (K,2)
sx = torch.exp(log_sigma_x).detach().numpy()            # sx: (K,)
sy = torch.exp(log_sigma_y).detach().numpy()            # sy: (K,)
th = thetas.detach().numpy()                            # th: (K,)
for k in range(K):
    px = (mu[k, 0] + 1) / 2 * (W - 1)                   # 中心 → 像素列
    py = (mu[k, 1] + 1) / 2 * (H - 1)                   # 中心 → 像素行
    e = Ellipse((px, py), width=2*sx[k]*(W-1)/2, height=2*sy[k]*(H-1)/2,
                angle=np.degrees(th[k]), fill=False, edgecolor="white", linewidth=0.6)
    axes[1, 1].add_patch(e)
axes[1, 1].set_title(f"训练后 + 椭圆高斯形状（{K} 个）")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("可视化图/Day34_各向异性高斯.png", dpi=120)
plt.show()
