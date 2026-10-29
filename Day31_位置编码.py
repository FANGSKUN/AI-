# Day 31：位置编码（Positional Encoding）—— NeRF 出清晰图的最后一块拼图
# 问题：MLP 天然偏「低频」（平滑），难拟合「高频」细节（锐利边缘、细纹理）
# 解法：把坐标编码成不同频率的 sin/cos，给 MLP 提供「傅里叶基」，让它能画高频

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 第 1 步：高频目标图（同心圆环）=================
def make_target(N=128):
    """一张「同心圆环」图：颜色随到中心的距离高频振荡（20 圈环 = 高频细节）"""
    xs = np.linspace(-1, 1, N)                       # xs: (N,) 一维坐标轴
    X, Y = np.meshgrid(xs, xs)                       # X, Y: (N,N) 坐标网格
    r = np.sqrt(X**2 + Y**2)                         # r: (N,N) 到原点距离
    target = 0.5 + 0.5 * np.sin(20 * np.pi * r)      # target: (N,N) 亮度值 [0,1]
    target = np.stack([target, target, target], axis=-1)  # target: (N,N,3) RGB 灰度图
    return target

# ================= 第 2 步：位置编码（关键，TODO）=================
def positional_encoding(coords, L=10):
    """coords (M, 2) → 高频编码 (M, 4L)
    TODO：对每个坐标，用 L 个频率（2^0, 2^1, ..., 2^{L-1}）的 sin 和 cos 编码。
    提示：
      对 i in range(L)：
        freq = 2**i * np.pi
        收集 sin(freq * coords) 和 cos(freq * coords)   （coords 形状 (M,2)）
      最后用 np.concatenate(列表, axis=-1) 拼起来
      可选：把原始 coords 也拼进去（NeRF 原文会保留原始坐标）
    返回 (M, 4L)（L=10 时为 (M, 40)）
    """
    # ---- 在这里填 ----
    enc = []
    for i in range(L):
        freq = 2**i * np.pi                          # freq: 标量，频率逐级翻倍
        enc.append(np.sin(freq * coords))            # sin: (M,2) 每个点的正弦
        enc.append(np.cos(freq * coords))            # cos: (M,2) 每个点的余弦
    # -------------------
    return np.concatenate(enc, axis=-1)              # 输出: (M, 4L) 高频指纹

# ================= 第 3 步：MLP =================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),    # in_dim → 128
            nn.Linear(hidden, hidden), nn.ReLU(),    # 128 → 128
            nn.Linear(hidden, 3),                    # 128 → 3（RGB）
        )
    def forward(self, x):                            # x: (M, in_dim)
        return torch.sigmoid(self.net(x))            # 输出: (M, 3) 颜色 [0,1]

# ================= 第 4 步：训练 =================
def train(model, coords_t, target_t, steps=1500):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        opt.zero_grad()
        pred = model(coords_t)                       # pred: (M, 3)
        loss = nn.MSELoss()(pred, target_t)          # loss: 标量
        loss.backward()
        opt.step()
    return loss.item()

def main():
    target = make_target(N=128)                      # target: (128,128,3)
    xs = np.linspace(-1, 1, 128)                     # xs: (128,)
    X, Y = np.meshgrid(xs, xs)                       # X, Y: (128,128)
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)  # coords: (16384,2) 所有像素点

    target_t = torch.from_numpy(target.reshape(-1, 3).astype(np.float32))  # (16384,3)

    # 模型 A：无位置编码（直接用坐标）
    model_raw = MLP(in_dim=2)
    coords_raw = torch.from_numpy(coords.astype(np.float32))  # coords_raw: (16384,2)
    loss_raw = train(model_raw, coords_raw, target_t)

    # 模型 B：有位置编码
    enc = positional_encoding(coords, L=10)          # enc: (16384,40) 高频指纹
    in_dim = enc.shape[-1]                           # in_dim = 40
    model_pe = MLP(in_dim=in_dim)
    coords_pe = torch.from_numpy(enc.astype(np.float32))  # coords_pe: (16384,40)
    loss_pe = train(model_pe, coords_pe, target_t)

    print(f"无位置编码 loss: {loss_raw:.6f}")
    print(f"有位置编码 loss: {loss_pe:.6f}")
    print(f"提升倍数: {loss_raw / loss_pe:.1f}x")

    with torch.no_grad():
        pred_raw = model_raw(coords_raw).numpy().reshape(128, 128, 3)  # (128,128,3) 无编码结果
        pred_pe = model_pe(coords_pe).numpy().reshape(128, 128, 3)     # (128,128,3) 有编码结果

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(target); axes[0].set_title("目标图（20 圈同心圆环，高频）")
    axes[1].imshow(pred_raw); axes[1].set_title(f"无位置编码 loss={loss_raw:.4f}")
    axes[2].imshow(pred_pe); axes[2].set_title(f"有位置编码 loss={loss_pe:.4f}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("可视化图/Day31_位置编码.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()
