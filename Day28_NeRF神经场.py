# Day 28：NeRF 入门——神经场（用 MLP 表示场景 = 隐式表示）
# 核心思想：NeRF 不"存"场景（点云/体素是显式存储），而是让 MLP"记住"规律：
#   输入坐标 (x,y,z) → 输出颜色 RGB + 密度 σ
# 今天用 2D 版直观感受：造一张图，让 MLP 从坐标 (x,y) 预测 RGB，拟合它。
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ================= 第 1 步：造目标图（要被拟合的"场景"）=================
def make_target(H, W):
    # 一张彩色图案：颜色随半径 r 和角度 ang 变化（平滑、函数式，MLP 学得动）
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    ang = np.arctan2(yy, xx)
    R = 0.5 + 0.5 * np.sin(r * 6)
    G = 0.5 + 0.5 * np.cos(ang * 3)
    B = 0.5 + 0.5 * np.sin(ang * 5 + r * 4)
    return np.stack([R, G, B], axis=-1)   # (H, W, 3)

# ================= 第 2 步：神经场（MLP：坐标 → RGB）=================
class NeuralField(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,3),
        )
        # TODO：3 层全连接 2 → hidden → hidden → 3，激活用 ReLU
        # 提示：nn.Linear, nn.ReLU；最后一层不要 ReLU（RGB 可以是任意值）

    def forward(self, coords):
        # TODO：coords (N,2) → RGB (N,3)
        return self.net(coords)

# ================= 第 3 步：训练（让 MLP 的渲染 ≈ 目标图）=================
def train(model, coords, target, steps=1500):
    # TODO：Adam 优化器 + MSE 损失，训练 steps 步
    # 提示：torch.optim.Adam(model.parameters(), lr=1e-3)
    #      每步：zero_grad → forward → loss → backward → step
    # 返回最终 loss（float）
    optm = torch.optim.Adam(model.parameters(), lr=0.001)
    for step in range(steps):
        optm.zero_grad()
        loss = nn.MSELoss()(model(coords), target)
        loss.backward()
        optm.step()
    return loss.item()

# ================= 第 4 步：主流程 + 可视化 =================
def main():
    H, W = 128, 128
    target = make_target(H, W)                      # (H,W,3)

    # 坐标网格 → (N,2)，RGB → (N,3)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)   # (H*W, 2)
    target_flat = target.reshape(-1, 3).astype(np.float32)                    # (H*W, 3)

    model = NeuralField().to(device)
    coords_t = torch.from_numpy(coords).to(device)
    target_t = torch.from_numpy(target_flat).to(device)

    loss = train(model, coords_t, target_t)
    print("最终 MSE:", round(loss, 6))

    # 渲染：用训练好的网络，对每个坐标算 RGB
    model.eval()
    with torch.no_grad():
        pred = model(coords_t).cpu().numpy().reshape(H, W, 3)
    pred = np.clip(pred, 0, 1)

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target); axes[0].set_title("目标图（要拟合的场景）"); axes[0].axis("off")
    axes[1].imshow(pred);    axes[1].set_title("神经场渲染（MLP 输出）");   axes[1].axis("off")
    plt.savefig("可视化图/Day28_神经场.png", dpi=150, bbox_inches="tight")
    print("已保存 可视化图/Day28_神经场.png")

if __name__ == "__main__":
    main()
