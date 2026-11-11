# Day 48 附加：看 Q 表在训练中怎么变（奖励扩散过程）
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0)

SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
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

Q = np.zeros((SIZE, SIZE, 4))
alpha, gamma, epsilon = 0.1, 0.9, 0.1

def choose_action(pos):
    return np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q[pos])

# 在这些轮数后拍快照
snap_points = [0, 5, 20, 100, 999]
snapshots = {}

for episode in range(1000):
    pos = start
    for t in range(200):
        a = choose_action(pos)
        nxt = step(pos, a)
        r = get_reward(pos, a)
        Q[pos[0], pos[1], a] += alpha * (r + gamma * np.max(Q[nxt[0], nxt[1]]) - Q[pos[0], pos[1], a])
        pos = nxt
        if pos == goal:
            break
    if episode in snap_points:
        snapshots[episode] = Q.copy()
        print(f"第 {episode:>4} 轮：Q 最大值 = {Q.max():8.3f}，Q>0 的格子数 = {(Q.max(axis=2) > 0).sum()}")

# ========== 可视化：每个阶段画一张 Q 表 ==========
arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))

for idx, (ep, Qsnap) in enumerate(snapshots.items()):
    ax = axes[idx]
    maxQ = Qsnap.max(axis=2)                 # 每格「最好动作」的价值
    best = Qsnap.argmax(axis=2)              # 每格「最好动作」是哪个方向
    im = ax.imshow(maxQ, cmap="Reds", vmin=0, vmax=100)
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) == goal:
                ax.text(c, r, "★", ha="center", va="center", fontsize=15, color="gold", weight="bold")
            elif (r, c) in obstacles:
                ax.text(c, r, "▓", ha="center", va="center", fontsize=17, color="black")
            elif maxQ[r, c] > 0:             # 还没学到（Q≤0）就画「·」
                ax.text(c, r, arrows[best[r, c]], ha="center", va="center", fontsize=13, weight="bold")
            else:
                ax.text(c, r, "·", ha="center", va="center", fontsize=12, color="gray")
    label = "训练前" if ep == 0 else f"第 {ep} 轮"
    ax.set_title(f"{label}\nQ>0 格子: {(maxQ > 0).sum()} 个")
    ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True, color="gray", lw=0.3)

fig.suptitle("Q-learning 训练过程：奖励从目标（★）一格一格扩散到全图", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("可视化图/Day48_Q变化过程.png", dpi=120)
plt.show()
