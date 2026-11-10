# Day 47：用世界模型做规划——机器人在脑海里「想象」找路
# 场景：8×8 小球世界，小球要从起点走到目标格子
# 世界模型（复用 Day 46 的 CNN）：f(当前帧, 动作) = 预测下一格
# 规划：每一步想象 4 个动作的后果，选离目标最近的 → 走一步 → 重复

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)
np.random.seed(0)

SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # 上 下 左 右
ACTION_NAMES = ["上", "下", "左", "右"]

def step(pos, a):
    r, c = pos
    dr, dc = ACTIONS[a]
    return (max(0, min(SIZE - 1, r + dr)), max(0, min(SIZE - 1, c + dc)))

# ========== 1. 训练 CNN 世界模型（复用 Day 46）==========
def make_input(pos, a):
    f = torch.zeros(1, SIZE, SIZE); f[0, pos[0], pos[1]] = 1.0
    act = torch.zeros(4, SIZE, SIZE); act[a] = 1.0
    return torch.cat([f, act], dim=0)                       # (5, 8, 8)

X, y = [], []
for _ in range(400):
    pos = (np.random.randint(SIZE), np.random.randint(SIZE))
    a = np.random.randint(4)
    nxt = step(pos, a)
    X.append(make_input(pos, a))
    y.append(nxt[0] * SIZE + nxt[1])
X = torch.stack(X)                                          # (400, 5, 8, 8)
y = torch.tensor(y)                                         # (400,)

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
    def forward(self, x):
        return self.net(x).reshape(x.shape[0], -1)          # (n, 64)

model = WorldModel()
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(500):
    logits = model(X)
    loss = crit(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
print("世界模型训练完成")

# ========== TODO 1：想象函数 ==========
# 用世界模型预测：从 pos 做动作 a，球会到哪一格
def imagine_next(pos, a):
    x = make_input(pos, a).unsqueeze(0)          # (1, 5, 8, 8)
    idx = model(x).argmax(dim=1).item()                                   # ---- 在这里填：model(x).argmax(dim=1).item() ----
    return (idx // SIZE, idx % SIZE)             # 格编号 → (r, c)

# ========== 规划（贪心）==========
goal = (7, 7)
start = (0, 0)

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

# ========== TODO 2：规划循环 ==========
# 每一步：想象 4 个动作的下一格，选离 goal 最近的那个动作，走一步，直到到达
path = [start]
pos = start
for _ in range(40):                              # 最多 40 步
    if pos == goal:
        break
    best_a, best_d = None, float("inf")
    for a in range(4):
        nxt = imagine_next(pos, a)               # 想象：按 a 会到哪
        d = manhattan(nxt,goal)                                # ---- 在这里填：算 nxt 到 goal 的距离 ----
        if d < best_d:
            best_d = d                        # ---- 在这里填：更新最近距离 ----
            best_a = a                        # ---- 在这里填：更新最优动作 ----
    pos = step(pos, best_a)                      # 真的走一步（真实世界）
    path.append(pos)

print(f"小球从 {start} 出发，目标 {goal}")
print(f"规划路径（{len(path)-1} 步）：")
for i, p in enumerate(path):
    print(f"  第{i}步: {p}" + ("  ← 到达目标！" if p == goal else ""))

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# 左：路径
ax = axes[0]
grid = np.zeros((SIZE, SIZE))
for (r, c) in path:
    grid[r, c] = 1
grid[goal[0], goal[1]] = 2
ax.imshow(grid, cmap="Blues", vmin=0, vmax=2)
for i, (r, c) in enumerate(path):
    ax.text(c, r, str(i), ha="center", va="center", fontsize=10, weight="bold", color="white")
ax.text(goal[1], goal[0], "★目标", ha="center", va="center", fontsize=13, color="red", weight="bold")
ax.set_title("世界模型规划出的路径（数字=步数）")
ax.set_xlabel("列 c"); ax.set_ylabel("行 r")
ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
ax.grid(True, color="gray", lw=0.5)

# 右：说明
ax = axes[1]
ax.axis("off")
steps = [
    "世界模型的用途：做「决策」",
    "",
    "每一步，机器人在脑海里想象：",
    "  按上→? 按下→? 按左→? 按右→?",
    "",
    "选「离目标最近」的那个动作",
    "→ 真的执行 → 重复",
    "",
    "= 想象 → 决策 → 行动 闭环",
]
ax.text(0.05, 0.5, "\n".join(steps), fontsize=12, va="center", family="sans-serif")
ax.set_title("用世界模型做规划（贪心）", fontsize=13)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("可视化图/Day47_世界模型规划.png", dpi=120)
plt.show()
