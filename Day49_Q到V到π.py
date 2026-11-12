# Day 49：从 Q 表到 V 函数和策略 π —— MDP 三件套的关系
# 核心：Q(s,a)「动作多好」、V(s)「格子多好」、π(s)「该走哪」
# 关系：V(s) = max_a Q(s,a)，π(s) = argmax_a Q(s,a)
# 验证：Bellman 最优方程 V*(s) = max_a [ r(s,a) + γ·V*(s') ]

import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0)

SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # 上 下 左 右
ACTION_NAMES = ["上", "下", "左", "右"]
start = (0, 0)
goal = (7, 7)
obstacles = {(2, 0), (3, 0), (4, 0)}

def step(pos, a):
    r, c = pos
    dr, dc = ACTIONS[a]
    nr, nc = max(0, min(SIZE - 1, r + dr)), max(0, min(SIZE - 1, c + dc))
    return pos if (nr, nc) in obstacles else (nr, nc)

def get_reward(pos, a):
    nxt = step(pos, a)
    if nxt == goal:
        return 100
    if nxt == pos:
        return -10
    return -1

# ========== 训练 Q-learning（复用 Day 48，得到一张训练好的 Q 表）==========
Q = np.zeros((SIZE, SIZE, 4))          # Q 表：(8, 8, 4)
alpha, gamma, epsilon = 0.1, 0.9, 0.3
def choose_action(pos):
    return np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q[pos])

for episode in range(3000):
    pos = start
    for t in range(500):
        a = choose_action(pos)
        nxt = step(pos, a)
        r = get_reward(pos, a)
        Q[pos[0], pos[1], a] += alpha * (r + gamma * np.max(Q[nxt[0], nxt[1]]) - Q[pos[0], pos[1], a])
        pos = nxt
        if pos == goal:
            break

print("Q-learning 训练完成，Q 表形状:", Q.shape)   # (8, 8, 4)

# ========== TODO 1：从 Q 表算 V(s) ==========
# V(s) = 这个格子「最好动作」的价值 = 4 个动作 Q 值里最大的那个
# 提示：对 axis=2（动作维）取 max，得到 (8,8)
V = Q.max(axis = 2)   # ---- 在这里填：Q.max(axis=2) ----

# ========== TODO 2：从 Q 表算策略 π(s) ==========
# π(s) = 这个格子「该走哪个方向」= 4 个动作 Q 值里最大的那个的下标
# 提示：对 axis=2 取 argmax，得到 (8,8) 的整数下标
pi = Q.argmax(axis = 2)   # ---- 在这里填：Q.argmax(axis=2) ----

# ========== TODO 3：验证 Bellman 最优方程 ==========
# V*(s) = max_a [ r(s,a) + γ·V*(s') ]，训练收敛后 V 应满足它
# 取一个格子 s，对 4 个动作算「奖励 + 折扣后的下一格价值」，取最大，应 ≈ V[s]
s = (5, 3)
rhs = max( get_reward(s, a) + gamma * V[step(s, a)]  for a in range(4) )  # ---- 在这里填：max( get_reward(s,a) + gamma*V[step(s,a)] for a in range(4) ) ----

print(f"\n验证 Bellman 最优方程：")
print(f"  V[{s}]（左边）= {V[s]:.3f}")
print(f"  max_a[r + γ·V(s')]（右边）= {rhs:.3f}")
print(f"  两者相差 = {abs(V[s] - rhs):.4f}（应接近 0）")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# 左：V 热力图（颜色 = 这个格子多好）
ax = axes[0]
im = ax.imshow(V, cmap="Reds", vmin=0, vmax=100)
for r in range(SIZE):
    for c in range(SIZE):
        if (r, c) == goal:
            ax.text(c, r, "★", ha="center", va="center", fontsize=15, color="gold", weight="bold")
        elif (r, c) in obstacles:
            ax.text(c, r, "▓", ha="center", va="center", fontsize=17, color="black")
        else:
            ax.text(c, r, f"{V[r,c]:.0f}", ha="center", va="center", fontsize=8, color="white")
ax.set_title("V(s)：每个格子多好（越红=越接近目标）")
ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
ax.grid(True, color="gray", lw=0.3)
fig.colorbar(im, ax=ax, fraction=0.046)

# 右：策略 π 箭头图（箭头 = 该往哪走）
ax = axes[1]
ax.imshow(np.zeros((SIZE, SIZE)), cmap="Greys", vmin=0, vmax=1)
for r in range(SIZE):
    for c in range(SIZE):
        if (r, c) == goal:
            ax.text(c, r, "★", ha="center", va="center", fontsize=15, color="gold", weight="bold")
        elif (r, c) in obstacles:
            ax.text(c, r, "▓", ha="center", va="center", fontsize=17, color="black")
        else:
            ax.text(c, r, arrows[pi[r, c]], ha="center", va="center", fontsize=14, weight="bold", color="steelblue")
ax.set_title("π(s)：每个格子该走哪个方向（策略）")
ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
ax.grid(True, color="gray", lw=0.3)

plt.tight_layout()
plt.savefig("可视化图/Day49_Q到V到π.png", dpi=120)
plt.show()
