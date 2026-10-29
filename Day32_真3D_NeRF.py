# Day 32：真 3D NeRF —— 隐式表示 + 体渲染 + 位置编码 三合一
# 把 Day 28（隐式表示）+ Day 29/30（体渲染）+ Day 31（位置编码）合进真正的 3D：
# 多个相机位姿看一个 3D 球体，训练 NeRF 学出整个 3D 场景，然后从「没见过的视角」合成新图

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：3D 场景（一个红球）=================
def scene_field(points):
    """points (..., 3) 3D 坐标 → sigma (...,), rgb (..., 3)
    TODO：球心原点、半径 0.5；球内 σ=10 红色，球外 σ=0 黑色。
    提示：
      dist = torch.norm(points, dim=-1)          # (...,) 到球心距离
      inside = dist < 0.5                        # (...,) 布尔
      sigma 用 torch.where(inside, 10.0, 0.0)
      rgb   用 torch.where(inside[...,None], 红, 黑)
    """
    # ---- 在这里填 ----
    dist = torch.norm(points, dim=-1)   # 算 dist → inside → sigma、rgb（见提示）
    inside = dist < 0.5
    sigma = torch.where(inside, 10.0, 0.0)
    rgb = torch.where(inside[...,None],torch.tensor([1.0, 0.0, 0.0]),torch.tensor([0.0, 0.0, 0.0]))
    # -------------------
    return sigma, rgb

# ================= 第 2 步：位置编码（torch 版，复用 Day 31）=================
def positional_encoding(coords, L):
    """coords (..., D) → (..., D*(2L+1))，含原始坐标
    TODO：L 个频率（2^0, 2^1, ..., 2^{L-1}）的 sin 和 cos，再加原始坐标。
    提示：复用 Day 31 的逻辑，但用 torch.sin/torch.cos，enc 列表先放 coords。
    """
    # ---- 在这里填 ----
    enc = []  # enc 列表：先放 coords，再循环 L 个频率加 sin、cos
    enc.append(coords)
    for i in range(L):
        freq = 2**i*np.pi
        enc.append(torch.sin(freq*coords))
        enc.append(torch.cos(freq*coords))
    # -------------------
    return torch.cat(enc, dim=-1)

# ================= 第 3 步：相机光线（3D）=================
def make_rays_3d(camera_pos, H=16, W=16, N=48, near=2.0, far=4.0):
    """相机在 camera_pos，看向原点，生成 H*W 条 3D 光线。
    每条光线沿方向采样 N 个点（t 从 near 到 far）。
    返回 coords (H*W, N, 3), dirs (H*W, 3), delta 标量"""
    camera_pos = camera_pos.float()                      # camera_pos: (3,)
    look_at = torch.tensor([0.0, 0.0, 0.0])              # look_at: (3,)
    forward = F.normalize(look_at - camera_pos, dim=0)   # forward: (3,) 看向原点
    world_up = torch.tensor([0.0, 1.0, 0.0])             # world_up: (3,)
    right = F.normalize(torch.cross(forward, world_up), dim=0)  # right: (3,)
    up = torch.cross(right, forward)                     # up: (3,)

    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # i,j: (H,W)
    x = (j.float() - (W - 1) / 2) / (W / 2)              # x: (H,W) ∈ [-1,1]
    y = (i.float() - (H - 1) / 2) / (H / 2)              # y: (H,W) ∈ [-1,1]

    dirs = (forward[None, None, :]                       # dirs: (H,W,3) 每条光线的方向
            + 0.5 * x[..., None] * right[None, None, :]
            + 0.5 * y[..., None] * up[None, None, :])
    dirs = F.normalize(dirs, dim=-1)                     # dirs: (H,W,3) 归一化

    ts = torch.linspace(near, far, N)                    # ts: (N,) 采样距离
    coords = camera_pos[None, None, :] + ts[None, :, None] * dirs[:, :, None, :]  # coords: (H,W,N,3)
    coords = coords.reshape(H * W, N, 3)                 # coords: (H*W, N, 3) 所有采样点
    delta = ts[1] - ts[0]                                # delta: 标量
    return coords, dirs.reshape(H * W, 3), delta         # dirs: (H*W, 3)

# ================= 第 4 步：体渲染（复用 Day 30，不改）=================
def volume_render(sigma, color, delta):
    """sigma (M,N)，color (M,N,3)，delta 标量 → C (M,3)"""
    alpha = 1 - torch.exp(-sigma * delta)                # alpha: (M,N)
    T_cum = torch.exp(-torch.cumsum(sigma * delta, dim=1))  # T_cum: (M,N)
    T = torch.cat([torch.ones_like(T_cum[:, :1]), T_cum[:, :-1]], dim=1)  # T: (M,N) 右移
    C = (T.unsqueeze(-1) * alpha.unsqueeze(-1) * color).sum(dim=1)  # C: (M,3)
    return C

# ================= 第 5 步：NeRF MLP（3D 坐标 + 方向）=================
class NeRF(nn.Module):
    def __init__(self, L_pos=6, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * (2 * L_pos + 1)                    # 位置编码维度 = 39
        dir_dim = 3 * (2 * L_dir + 1)                    # 方向编码维度 = 27
        self.net = nn.Sequential(
            nn.Linear(pos_dim + dir_dim, hidden), nn.ReLU(),   # 66 → 128
            nn.Linear(hidden, hidden), nn.ReLU(),              # 128 → 128
            nn.Linear(hidden, 4),                              # 128 → 4
        )

    def forward(self, x, d):
        """x (..., N, 3) 位置, d (..., 3) 方向 → sigma (..., N), rgb (..., N, 3)
        TODO：
          1. x 过位置编码 → x_enc (..., N, pos_dim)
          2. d 过位置编码 → d_enc (..., dir_dim)，再 unsqueeze+expand 广播到 (..., N, dir_dim)
          3. cat(x_enc, d_enc, dim=-1) → 过 net → out (..., N, 4)
          4. σ = softplus(out[...,0])，rgb = sigmoid(out[...,1:])
        提示：方向编码广播：d_enc.unsqueeze(-2).expand(*x.shape[:-1], -1)
        """
        # ---- 在这里填 ----
        x_enc,d_enc = positional_encoding(x,self.L_pos),positional_encoding(d,self.L_dir)  # 位置编码 x、d → cat → net → softplus 出 σ、sigmoid 出 rgb
        d_enc = d_enc.unsqueeze(-2).expand(*x.shape[:-1], -1)
        out = self.net(torch.cat((x_enc, d_enc), dim=-1))
        sigma = F.softplus(out[...,0])
        rgb = torch.sigmoid(out[...,1:])
        # -------------------
        return sigma, rgb

# ================= 第 6 步：训练 + 新视角合成 =================
def render_view(model, camera_pos, H=16, W=16, N=48):
    """给定相机位姿，用 NeRF 渲染一张图 (H,W,3)。也用真实场渲染 target 对比。"""
    coords, dirs, delta = make_rays_3d(camera_pos, H=H, W=W, N=N)   # coords:(M,N,3) dirs:(M,3)
    sigma_true, color_true = scene_field(coords)          # 真实场
    target = volume_render(sigma_true, color_true, delta) # target: (M,3)
    with torch.no_grad():
        sigma, rgb = model(coords, dirs)                  # NeRF 预测
        pred = volume_render(sigma, rgb, delta)           # pred: (M,3)
    return target.reshape(H, W, 3), pred.reshape(H, W, 3)  # 各 (H,W,3)

def main():
    # 训练视角：绕 y 轴 4 个角度，半径 3，高度 0.4
    train_angles = [0, 45, 90, 135, 180, 225, 270]
    train_rays = []
    for angle in train_angles:
        rad = angle * np.pi / 180
        camera_pos = torch.tensor([3 * np.cos(rad), 0.4, 3 * np.sin(rad)], dtype=torch.float32)
        coords, dirs, delta = make_rays_3d(camera_pos)
        sigma_true, color_true = scene_field(coords)
        target = volume_render(sigma_true, color_true, delta)
        train_rays.append((coords, dirs, delta, target))  # 每个视角一份 (光线, 方向, 间距, 真值)

    model = NeRF(L_pos=6, L_dir=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("开始训练（多视角 3D NeRF）...")
    for step in range(6000):
        coords, dirs, delta, target = train_rays[step % len(train_rays)]  # 轮流训练 4 个视角
        opt.zero_grad()
        sigma, rgb = model(coords, dirs)
        pred = volume_render(sigma, rgb, delta)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss.item():.6f}")

    # 可视化：左上/右上 = 训练视角(0°)真实 vs NeRF；左下/右下 = 新视角(45°)真实 vs NeRF
    cam0 = torch.tensor([3.0, 0.4, 0.0])                 # 训练视角 0°
    cam22 = torch.tensor([3 * np.cos(np.pi/8), 0.4, 3 * np.sin(np.pi/8)])  # 新视角 45°（没训练过）
    t0, p0 = render_view(model, cam0)
    t22, p22 = render_view(model, cam22)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(t0); axes[0, 0].set_title("训练视角 0°（真实场）")
    axes[0, 1].imshow(p0); axes[0, 1].set_title("训练视角 0°（NeRF）")
    axes[1, 0].imshow(t22); axes[1, 0].set_title("新视角 22.5°（真实场）")
    axes[1, 1].imshow(p22); axes[1, 1].set_title("新视角 22.5°（NeRF 合成）")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("可视化图/Day32_真3D_NeRF.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()
