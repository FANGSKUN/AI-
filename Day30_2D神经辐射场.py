# Day 30：2D 神经辐射场——隐式表示 + 体渲染（NeRF 完整闭环）
# 把 Day 28 的「隐式表示」和 Day 29 的「体渲染」合起来：
# 一个 MLP 表示 2D 场景的密度场 + 颜色场，体渲染出图像，训练让渲染匹配真实场景

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：生成光线（相机在 x=-1，沿 +x 看）=================
def make_rays(H=64, N=64):
    """发 H 条水平光线（y 从 -0.5 到 0.5），每条沿 x 从 -1 到 1 采样 N 个点。
    返回 coords (H,N,2) 和采样间距 delta。"""
    ys = np.linspace(-0.5, 0.5, H)
    xs = np.linspace(-1.0, 1.0, N)
    delta = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, ys)            # 各 (H, N)
    coords = np.stack([X, Y], axis=-1).astype(np.float32)   # (H, N, 2)
    return coords, delta

# ================= 第 2 步：体渲染（批量版，复用 Day 29 公式）=================
def volume_render(sigma, color, delta):
    """sigma (H,N)，color (H,N,3)，delta 标量 → C (H,3)
    TODO：把 Day 29 的三行公式改成「沿 dim=1（N 维）」的批量版。
    提示：
      alpha：1 - exp(-sigma * delta)，sigma 形状 (H,N)
      T：先 torch.cumsum(sigma*delta, dim=1)（沿 N 维累积），再 exp，再右移
         （cat 一个 ones 在最前面，砍掉最后一项，都沿 dim=1）
      C：T、alpha 用 unsqueeze(-1) 变 (H,N,1)，乘 color (H,N,3)，再 sum(dim=1)
    """
    # ---- 在这里填 ----
    alpha = 1 - torch.exp(-sigma * delta)
    T_cum = torch.exp(-torch.cumsum(sigma * delta, dim=1))
    T = torch.cat([torch.ones_like(T_cum[:, :1]), T_cum[:, :-1]], dim=1)
    C = (T.unsqueeze(-1) * alpha.unsqueeze(-1) * color).sum(dim=1)
    # -------------------
    return C

# ================= 第 3 步：神经场（MLP 表示密度场 + 颜色场）=================
class NeuralRadianceField(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # TODO：MLP 输入 2 → 输出 4（前 1 维 σ，后 3 维 RGB）
        self.net = nn.Sequential(
            nn.Linear(2, hidden),nn.ReLU(),
            nn.Linear(hidden, hidden),nn.ReLU(),
            nn.Linear(hidden, 4)
        )
        # 提示：nn.Linear(2,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,4)

    def forward(self, coords):
        # coords (H,N,2) → out (H,N,4)
        # TODO：拆出 σ 和 RGB
        #   σ：密度必须 > 0，用 softplus：F.softplus(out[..., 0])
        #   rgb：颜色在 [0,1]，用 sigmoid：torch.sigmoid(out[..., 1:])
        out = self.net(coords)
        sigma = F.softplus(out[..., 0])
        rgb = torch.sigmoid(out[..., 1:])
        return sigma, rgb

# ================= 第 4 步：训练 + 验证 =================
def main():
    coords, delta = make_rays(H=64, N=64)
    X = coords[..., 0]; Y = coords[..., 1]

    # 真实场景：红色圆盘（中心 (0,0) 半径 0.4，盘内 σ=5 红色，盘外空）
    r = np.sqrt(X**2 + Y**2)
    inside = (r < 0.4).astype(np.float32)         #这里的inside就是一个(64,64)的布尔型数组
    sigma_true = inside * 5.0                                         # (H,N)
    color_true = np.stack([inside, np.zeros_like(inside), np.zeros_like(inside)], axis=-1)  # (H,N,3)

    # ground truth 图像：用真实场体渲染
    target = volume_render(torch.from_numpy(sigma_true), torch.from_numpy(color_true), delta)

    model = NeuralRadianceField(hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    coords_t = torch.from_numpy(coords)

    print("开始训练（让 MLP 学出圆盘的密度场 + 颜色场）...")
    for step in range(3000):
        opt.zero_grad()
        sigma, rgb = model(coords_t)
        pred = volume_render(sigma, rgb, delta)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"  step {step:4d}  loss {loss.item():.6f}")

    with torch.no_grad():
        sigma, rgb = model(coords_t)
        pred = volume_render(sigma, rgb, delta)
    print(f"最终 loss {F.mse_loss(pred, target).item():.6f}")

    # 可视化：左=真实渲染，中=神经场渲染，右=学到的密度场热图
    ys = np.linspace(-0.5, 0.5, 64)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(target.numpy()[None, :, :], aspect="auto", extent=[0, 1, -0.5, 0.5])
    axes[0].set_title("真实渲染（红圆盘投影）")
    axes[0].set_xlabel("像素")
    axes[0].set_ylabel("光线 y")
    axes[1].imshow(pred.numpy()[None, :, :], aspect="auto", extent=[0, 1, -0.5, 0.5])
    axes[1].set_title("神经场渲染")
    axes[1].set_xlabel("像素")
    gx = np.linspace(-1, 1, 100); gy = np.linspace(-0.5, 0.5, 50)
    GX, GY = np.meshgrid(gx, gy)
    gcoords = torch.from_numpy(np.stack([GX, GY], axis=-1)).float()
    with torch.no_grad():
        gsigma, _ = model(gcoords)
    axes[2].imshow(gsigma.numpy(), origin="lower", extent=[-1, 1, -0.5, 0.5], cmap="hot")
    axes[2].set_title("学到的密度场 σ(x,y)")
    axes[2].set_xlabel("x")
    plt.tight_layout()
    plt.savefig("可视化图/Day30_2D神经辐射场.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()
