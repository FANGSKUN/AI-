# Day 48：强化学习 Q-learning——小球学会绕障碍、长远规划
# 核心：Q(s,a) = 在位置 s 做动作 a 的「长远价值」
# 训练 = 试错：撞墙降 Q、到目标升 Q，慢慢学出一张「经验地图」
# 更新公式（Bellman）：Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0)

SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # 上 下 左 右
ACTION_NAMES = ["上", "下", "左", "右"]

start = (0, 0)
goal = (7, 7)
# 障碍物：第 0 列的第 2~4 行（一堵竖墙，挡住贪心「先向下」的路）
obstacles = {(2, 0), (3, 0), (4, 0)}

def step(pos, a):
    r, c = pos
    dr, dc = ACTIONS[a]
    nr, nc = max(0, min(SIZE - 1, r + dr)), max(0, min(SIZE - 1, c + dc))
    if (nr, nc) in obstacles:
        return pos        # 撞障碍：原地不动
    return (nr, nc)

# ========== TODO 1：奖励函数 ==========
# 返回：到目标 +100，撞障碍（还在原地且非目标）-10，其他 -1
def get_reward(pos, a):
    nxt = step(pos, a)
    if nxt == goal:
        return 100       # ---- 在这里填：+100 ----
    if nxt == pos:
        return -10       # ---- 在这里填：撞障碍，-10 ----
    return -1             # 正常走一步

# ========== Q-learning 训练 ==========
Q = np.zeros((SIZE, SIZE, 4))     # Q 表：64 位置 × 4 动作，全 0 开始
alpha = 0.1                        # 学习率
gamma = 0.9                        # 折扣因子（看重未来）
epsilon = 0.1                      # 探索率

def choose_action(pos):
    # ========== TODO 2：ε-贪心选动作 ==========
    # 10% 概率随机探索，90% 概率选 Q 值最大的动作
    if np.random.rand() < epsilon:
        return np.random.randint(4)              # 探索
    return np.argmax(Q[pos])   # ---- 在这里填：选 Q[pos] 最大的动作下标 ----

steps_history = []
for episode in range(1000):
    pos = start
    for t in range(200):
        a = choose_action(pos)
        nxt = step(pos, a)
        r = get_reward(pos, a)
        # ========== TODO 3：Q 值更新（Bellman 方程）==========
        # Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        Q[pos[0], pos[1], a] = Q[pos[0], pos[1], a]+alpha*(r+gamma*np.max(Q[nxt[0], nxt[1]]) -  Q[pos[0], pos[1], a])# ---- 在这里填：完整更新公式 ----
        pos = nxt
        if pos == goal:
            break
    steps_history.append(t + 1)

print(f"训练完成。最后 100 轮平均步数 = {np.mean(steps_history[-100:]):.1f}")

# ========== 训练后用学到的 Q 表走一遍（纯利用，不探索）==========
path = [start]
pos = start
for _ in range(100):
    if pos == goal:
        break
    a = int(np.argmax(Q[pos[0], pos[1]]))          # 每步选 Q 最大的动作
    pos = step(pos, a)
    path.append(pos)

print(f"\nQ-learning 规划的路径（{len(path)-1} 步）：")
for i, p in enumerate(path):
    print(f"  第{i}步: {p}" + ("  ← 到达目标！" if p == goal else ""))

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

# 左：训练曲线
ax = axes[0]
ax.plot(steps_history, lw=1, color="steelblue", alpha=0.6)
ax.plot(np.convolve(steps_history, np.ones(50)/50, mode="valid"), lw=2, color="crimson")
ax.set_title("每轮步数（红线=50轮滑动平均）\n越走越短 = 学到规律了")
ax.set_xlabel("episode"); ax.set_ylabel("步数")
ax.grid(True, alpha=0.3)

# 右：路径 + 障碍
ax = axes[1]
grid = np.zeros((SIZE, SIZE))
for o in obstacles:
    grid[o] = 0.5
for (r, c) in path:
    grid[r, c] = 1
grid[goal] = 2
ax.imshow(grid, cmap="Blues", vmin=0, vmax=2)
for o in obstacles:
    ax.text(o[1], o[0], "▓", ha="center", va="center", fontsize=16, color="black")
for i, (r, c) in enumerate(path):
    ax.text(c, r, str(i), ha="center", va="center", fontsize=9, weight="bold", color="white")
ax.text(goal[1], goal[0], "★", ha="center", va="center", fontsize=14, color="red", weight="bold")
ax.set_title("Q-learning 绕开障碍找到路")
ax.set_xlabel("列 c"); ax.set_ylabel("行 r")
ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
ax.grid(True, color="gray", lw=0.5)

plt.tight_layout()
plt.savefig("可视化图/Day48_Q学习.png", dpi=120)
plt.show()
