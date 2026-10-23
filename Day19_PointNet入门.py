# -*- coding: utf-8 -*-
r"""
Day19 项目：PointNet 入门——对称函数解决点云「无序性」（Phase 4 核心第一课）
=========================================================================
点云有「三座大山」：无序 / 稀疏不规则 / 刚体变换。今天攻第一座——无序。

问题：同一组点有 N! 种排列，都是同一个物体。网络不能因为点的顺序变了就
改口。PointNet（2017 CVPR）的解法就一句话：
    「最后用对称函数（max），输出就不在乎点的顺序。」

对称函数：f([a,b,c]) = f([c,a,b]) = f([b,a,c])，任意排列结果不变。
max / sum / mean 都是对称函数；"按顺序处理"（RNN、顺序 MLP）不是。

今天动手：训练一个极简 PointNet，区分「球面点云 vs 立方体面点云」，
并亲手验证「打乱点的顺序，输出不变」。

运行：
    conda activate ai
    python Day19_自检.py
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


# ============ 1. 造两类点云数据 ============

def make_sphere(n, r=1.0):
    """球面点云 (n, 3)。昨天 Day18 写过。"""
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def make_cube_surface(n, half=1.0):
    """立方体表面点云 (n, 3)：每个点随机落在 6 个面之一。

    思路：6 个面是 ±x、±y、±z。随机选一个面，再在这个面（2D 正方形）上
    均匀取一个点。比如 "+z 面" 就是 z 固定 = half，x、y 在 [-half, half] 里随机。
    """
    points = np.zeros((n, 3))
    for i in range(n):
        face = np.random.randint(0, 6)              # 随机选面
        u = np.random.uniform(-half, half)          # 面上的一个坐标
        v = np.random.uniform(-half, half)          # 面上的另一个坐标
        # TODO 1：按 face 填这个点的 (x, y, z)。
        #   0 → +z 面；1 → -z 面；2 → +y 面；3 → -y 面；4 → +x 面；5 → -x 面
        if face == 0:
            points[i] = [u, v, half]
        elif face == 1:
            points[i] = [u, v, -half]
        elif face == 2:
            points[i] = [u, half, v]
        elif face == 3:
            points[i] = [u, -half, v]
        elif face == 4:
            points[i] = [half, u, v]
        else:
            points[i] = [-half, u, v]
    return points


# ============ 2. 极简 PointNet ============

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 每个点独立过共享 MLP：3 维坐标 → 128 维特征
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        # 全局特征 128 → 类别
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3) —— B 个点云，每个 N 个点，每个点 3 个坐标
        feat = self.mlp(x)                 # (B, N, 128)  每个点独立提特征
        # TODO 2：对称函数——对「点」那一维取 max，压掉顺序。
        #   提示：B=第0维、N=第1维、特征=第2维，要压掉的是「点」这一维，dim = ?
        global_feat = feat.max(dim=1)[0]   # (B, 128)
        return self.classifier(global_feat)            # (B, num_classes)


# ============ 3. 验证「无序性」：打乱点顺序，输出应不变 ============

def permutation_diff(model, points):
    """返回「原顺序输出」和「打乱顺序输出」的最大差值。应该 ≈ 0。

    这是 PointNet 的灵魂检验：对称函数让输出对点的排列不敏感。
    """
    model.eval()
    x = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)
    n = x.shape[1]
    perm = torch.randperm(n)             # 随机排列索引
    x_shuffled = x[:, perm, :]           # 按 perm 打乱「点」的顺序
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x_shuffled)
    # TODO 3：算两个输出的「最大差值」。越接近 0 越说明"顺序无关"。
    #   提示：用 .abs() 取绝对值，再 .max() 取最大，最后 .item() 转成 python 数
    diff = (out1 - out2).abs().max().item()
    return diff


# ============ 4. 训练：区分球 vs 立方体 ============

def make_dataset(n_each=200, n_points=512):
    """造训练数据：n_each 个球 + n_each 个立方体，标签 0/1。"""
    spheres = np.stack([make_sphere(n_points) for _ in range(n_each)])
    cubes = np.stack([make_cube_surface(n_points) for _ in range(n_each)])
    X = np.concatenate([spheres, cubes], axis=0)          # (2*n_each, N, 3)
    y = np.array([0] * n_each + [1] * n_each)             # 球=0，立方体=1
    # 打乱样本顺序
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def train(model, X, y, epochs=20, lr=0.01):
    """训练循环，返回每轮的准确率列表。"""
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    accs = []
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(Xt)               # (2*n_each, 2)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        pred = logits.argmax(dim=1)
        acc = (pred == yt).float().mean().item()
        accs.append(acc)
        if epoch == 0 or (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:2d}  loss={loss.item():.4f}  acc={acc:.3f}")
    return accs


def draw_loss(accs, filename="可视化图/Day19_训练曲线.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(accs) + 1), accs, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("准确率")
    ax.set_title("PointNet 区分球 vs 立方体")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    return filename


def main():
    print(f"设备：{device}")
    model = PointNet().to(device)

    # 验证无序性：一个随机点云，打乱顺序，输出差应 ≈ 0
    sample = make_sphere(512)
    diff = permutation_diff(model, sample)
    print(f"打乱点顺序后输出最大差 = {diff:.6f}（≈0 说明顺序无关）")

    # 训练
    print("造数据 + 训练中...")
    X, y = make_dataset(n_each=200, n_points=512)
    accs = train(model, X, y, epochs=20)
    print(f"最终准确率 = {accs[-1]:.3f}")
    draw_loss(accs)


if __name__ == "__main__":
    main()
