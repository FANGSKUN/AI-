# Day 46 对比实验：CNN vs MLP 的泛化能力
# 同样的训练数据（400 个随机样本，只覆盖 204/256 种组合），
# 分别用 MLP（全连接）和 CNN（卷积）学小球移动，
# 然后全量测试全部 256 种「位置×动作」组合，看谁更能举一反三

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)
np.random.seed(0)

SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = ["上", "下", "左", "右"]

def step(pos, a):
    r, c = pos
    dr, dc = ACTIONS[a]
    return (max(0, min(SIZE - 1, r + dr)), max(0, min(SIZE - 1, c + dc)))

# ===== 造同样的训练数据 =====
def make_mlp_input(pos, a):
    """MLP 输入：帧展平 64 维 + 动作 one-hot 4 维 = 68 维"""
    f = torch.zeros(SIZE, SIZE); f[pos[0], pos[1]] = 1.0
    act = torch.zeros(4); act[a] = 1.0
    return torch.cat([f.reshape(-1), act])                 # (68,)

def make_cnn_input(pos, a):
    """CNN 输入：帧 1 通道 + 动作 4 通道（该动作通道全 1）= 5 通道 8×8"""
    f = torch.zeros(1, SIZE, SIZE); f[0, pos[0], pos[1]] = 1.0
    act = torch.zeros(4, SIZE, SIZE); act[a] = 1.0
    return torch.cat([f, act], dim=0)                      # (5, 8, 8)

X_mlp, X_cnn, y = [], [], []
for _ in range(400):
    pos = (np.random.randint(SIZE), np.random.randint(SIZE))
    a = np.random.randint(4)
    nxt = step(pos, a)
    X_mlp.append(make_mlp_input(pos, a))
    X_cnn.append(make_cnn_input(pos, a))
    y.append(nxt[0] * SIZE + nxt[1])
X_mlp = torch.stack(X_mlp)                                 # (400, 68)
X_cnn = torch.stack(X_cnn)                                 # (400, 5, 8, 8)
y = torch.tensor(y)                                        # (400,)

# ===== 两个模型 =====
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(68, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, x):
        return self.net(x)                                  # (n, 64)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),     # 5 通道 → 32 通道
            nn.Conv2d(32, 1, 3, padding=1),                 # 32 通道 → 1 通道（预测下一帧）
        )
    def forward(self, x):
        return self.net(x).reshape(x.shape[0], -1)          # (n, 64)

# ===== 训练函数 =====
def train(model, X, epochs=500):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = []
    for _ in range(epochs):
        logits = model(X)
        loss = crit(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
    return hist[-1], hist

print("训练中……")
mlp, cnn = MLP(), CNN()
loss_mlp, hist_mlp = train(mlp, X_mlp)
loss_cnn, hist_cnn = train(cnn, X_cnn)
print(f"MLP 最终 loss = {loss_mlp:.4f}   |   CNN 最终 loss = {loss_cnn:.4f}")

# ===== 全量测试：遍历所有 256 种组合 =====
def test_all(model, make_input):
    correct, wrong = 0, []
    for r in range(SIZE):
        for c in range(SIZE):
            for a in range(4):
                pos = (r, c)
                x = make_input(pos, a).unsqueeze(0)
                pred = model(x).argmax(dim=1).item()
                true_idx = step(pos, a)[0] * SIZE + step(pos, a)[1]
                if pred == true_idx:
                    correct += 1
                else:
                    wrong.append((pos, a))
    return correct, wrong

c_mlp, wrong_mlp = test_all(mlp, make_mlp_input)
c_cnn, wrong_cnn = test_all(cnn, make_cnn_input)
total = SIZE * SIZE * 4
print(f"\n全量测试 256 种组合：")
print(f"  MLP: {c_mlp}/{total} 正确  （错 {total-c_mlp} 个，都是训练没见过的）")
print(f"  CNN: {c_cnn}/{total} 正确  （错 {total-c_cnn} 个）")

# ===== 可视化：正确率对比 + 训练曲线 =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
bars = ax.bar(["MLP（全连接）", "CNN（卷积）"],
              [c_mlp / total * 100, c_cnn / total * 100],
              color=["tomato", "steelblue"], width=0.5)
ax.set_ylim(0, 105)
ax.set_ylabel("全量测试正确率 %")
ax.set_title("同样 400 个训练样本，谁更能举一反三？")
for b, v in zip(bars, [c_mlp, c_cnn]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{v}/256", ha="center", fontsize=12)
ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.5)
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
ax.plot(hist_mlp, label="MLP", color="tomato", lw=1.5)
ax.plot(hist_cnn, label="CNN", color="steelblue", lw=1.5)
ax.set_xlabel("epoch"); ax.set_ylabel("训练 loss")
ax.set_title("两者训练 loss 都降得很低")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("可视化图/Day46_CNN对比MLP.png", dpi=120)
plt.show()
