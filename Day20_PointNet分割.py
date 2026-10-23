# -*- coding: utf-8 -*-
r"""
Day20 项目：点云分割（PointNet 分割分支 + 跳跃连接，Phase 4 继续）
=================================================================
昨天分类是「整块点云 → 1 个标签」；今天分割是「整块点云 → 每个点 1 个标签」。

输出形状的区别：
  分类： (B, C)       一个点云一个标签
  分割： (B, N, C)    每个点一个标签

为什么昨天的 max pooling 不够？因为它把「点」那一维压掉，丢了「哪个点在哪」。
分割要每个点的标签，必须保留每点信息 → 用「跳跃连接」把全局特征复印回每个点。

今天任务：生成一个「哑铃」点云（两个球心在 ±d 的球拼一起），训练 PointNet 分割
分支，给每个点分类「属于球 0 还是球 1」。

运行：
    conda activate ai
    python Day20_自检.py
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ============ 1. 造「哑铃」分割数据 ============

def make_sphere(n, r=1.0):
    """球面点云 (n, 3)。"""
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def make_dumbbell(n, r=1.0, d=0.6):
    """哑铃点云：两个球（球心在 z=+d 和 z=-d）拼一起，每个点标「属于哪个球」。

    返回 (points, labels)：
      points: (n, 3)
      labels: (n,)   0=上面的球(+d)，1=下面的球(-d)
    """
    n0 = n // 2
    n1 = n - n0
    # TODO 1：生成两个球面点云，分别平移到球心 (0,0,+d) 和 (0,0,-d)。
    #   提示：make_sphere 返回球心在原点的球，加一个偏移向量 [0,0,±d] 即可平移。
    #         numpy 广播：points + np.array([0, 0, d]) 会把每个点都加上这个偏移。
    p0 = make_sphere(n0, r) + np.array([0, 0, d])   # 上面的球
    p1 = make_sphere(n1, r) + np.array([0, 0, -d])   # 下面的球
    points = np.concatenate([p0, p1], axis=0)          # (n, 3)
    labels = np.array([0] * n0 + [1] * n1)             # (n,)
    return points, labels


def make_dataset(n_each=100, n_points=256):
    """造一批哑铃点云：n_each 个，每个 n_points 个点，返回 (X, y)。

    X: (n_each, N, 3)   y: (n_each, N)   每点一个标签
    """
    Xs, ys = [], []
    for _ in range(n_each):
        pts, lab = make_dumbbell(n_points)
        Xs.append(pts)
        ys.append(lab)
    return np.stack(Xs), np.stack(ys)


# ============ 2. PointNet 分割分支 ============

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 每点独立提特征（和分类一样）
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        # 全局特征升维
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
        )
        # 每点分类头：输入 = 每点特征(128) + 全局特征(256) = 384
        self.classifier = nn.Sequential(
            nn.Linear(128 + 256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        feat = self.mlp(x)                       # (B, N, 128)  每点特征

        # TODO 2：max pooling 得到全局特征（和昨天一样，压掉「点」那一维）
        global_feat = feat.max(dim=1)[0]   # (B, 128)

        global_feat = self.global_mlp(global_feat)     # (B, 256)

        # TODO 3：跳跃连接——把全局特征「复制 N 份」，拼到每点特征后面。
        #   第一步：unsqueeze(1) 把 (B,256) 变成 (B,1,256)
        #   第二步：expand(B, N, 256) 把第 1 维从 1 扩到 N（复制 N 份，不占新内存）
        global_feat = global_feat.unsqueeze(1).expand(B, N, 256)

        #   第三步：沿最后一维拼接 → (B, N, 128+256=384)
        combined = torch.cat([feat, global_feat], dim=2)

        return self.classifier(combined)         # (B, N, num_classes)


# ============ 3. 训练 + 可视化 ============

def train(model, X, y, epochs=40, lr=0.01):
    """训练分割模型。X:(n_each,N,3) y:(n_each,N)，返回每轮逐点准确率。"""
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    accs = []
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(Xt)                       # (n_each, N, 2)
        # 逐点算 loss：把 (B,N,C) 展平成 (B*N,C)，(B,N) 展平成 (B*N,)
        loss = loss_fn(logits.reshape(-1, 2), yt.reshape(-1))
        loss.backward()
        opt.step()
        pred = logits.argmax(dim=-1)             # (B, N) 每个点的预测类别
        acc = (pred == yt).float().mean().item()
        accs.append(acc)
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:2d}  loss={loss.item():.4f}  逐点acc={acc:.3f}")
    return accs


def visualize(pts, y_true, y_pred, filename="可视化图/Day20_分割.png"):
    """3D 散点对比：真实标签 vs 预测标签（红色=球0，蓝色=球1）。"""
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=y_true, s=8, cmap="coolwarm")
    ax1.set_title("真实标签")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=y_pred, s=8, cmap="coolwarm")
    ax2.set_title("预测标签")
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    return filename


def main():
    print(f"设备：{device}")
    model = PointNetSeg().to(device)

    print("造哑铃数据 + 训练中...")
    X, y = make_dataset(n_each=100, n_points=256)
    accs = train(model, X, y, epochs=40)
    print(f"最终逐点准确率 = {accs[-1]:.3f}")

    # 可视化一个样例的预测 vs 真实
    model.eval()
    sample = X[0]
    with torch.no_grad():
        logits = model(torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device))
        pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
    visualize(sample, y[0], pred)


if __name__ == "__main__":
    main()
