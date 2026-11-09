# Day 45：世界模型（World Model）—— 让模型学会「预测未来」
# 核心思想：世界模型 = f(当前状态, 动作) = 预测下一状态
# 机器人场景：状态=摄像头图像（ViT 变 token），动作=马达指令，世界模型回答「我这么做，世界会变成什么样」
# 为什么叫「世界模型」：它内部编码了世界的运转规律，能像「做梦」一样想象未来

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 世界：8×8 网格 + 一个小球 =================
SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # 0上 1下 2左 3右
ACTION_NAMES = ["上", "下", "左", "右"]

# ================= TODO 1：世界规则 step 函数 =================
def step(pos, action):
    """给定小球位置 (r, c) 和动作，返回新位置 (r', c')
    规则：按动作移动一格，但碰边界不动（clamp 在 0 ~ SIZE-1）
    提示：new_r = max(0, min(SIZE-1, r + dr))
    """
    r, c = pos
    dr, dc = ACTIONS[action]
    new_r = max(0,min(SIZE-1,r+dr))   # ---- 在这里填 ----
    new_c = max(0,min(SIZE-1,c+dc))   # ---- 在这里填 ----
    return (new_r, new_c)

# ================= TODO 2：模拟一段轨迹（真实世界怎么运转）=================
# 小球从 (4,4) 出发，做一串动作，记录每一帧位置
start = (4, 4)
action_seq = [3, 1, 3, 3, 2, 0, 0]   # 右、下、右、右、左、上、上
trajectory = [start]
pos = start# ---- 在这里填：循环 step 累积轨迹 ----
for a in action_seq:
    pos = step(pos,a)
    trajectory.append(pos)
# （补全：for a in action_seq: pos = step(pos, a); trajectory.append(pos)）

# ================= TODO 3：把「帧」变成 token（世界模型怎么看世界）=================
# 一帧 = 8×8 网格，小球位置是 1、其余 0；展平成 64 维向量 = 一个「状态 token」
def frame_to_token(pos):
    frame = torch.zeros(SIZE, SIZE)    # frame: (8,8)
    frame[pos[0], pos[1]] = 1.0        # 小球位置置 1
    return frame.reshape(64)   # ---- 在这里填：把 (8,8) 帧展平成 64 维向量（提示：reshape）----

# 展示第一帧和最后一帧的 token
first_token = frame_to_token(trajectory[0])
last_token = frame_to_token(trajectory[-1])

# ================= 打印 =================
print("小球轨迹（真实世界）：")
for i, p in enumerate(trajectory):
    print(f"  第{i}帧: 位置 {p}  (动作 {ACTION_NAMES[action_seq[i]] if i < len(action_seq) else '止'})")
print("\n第一帧 token（64 维，1 的位置 = 小球所在格）：")
print(first_token.numpy().reshape(8, 8).astype(int))
print("\n最后一帧 token：")
print(last_token.numpy().reshape(8, 8).astype(int))
print("\n世界模型要学的东西：给定「当前帧 token + 动作」，预测「下一帧 token」")

# ================= 可视化 =================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左：小球轨迹
ax = axes[0]
grid = np.zeros((SIZE, SIZE))
for (r, c) in trajectory:
    grid[r, c] = 1
ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
for i, (r, c) in enumerate(trajectory):
    ax.text(c, r, str(i), ha="center", va="center", fontsize=12, weight="bold",
            color="white" if i == 0 else "black")
ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
ax.set_title("小球轨迹（数字 = 帧序号）\n世界规则：按动作移动，碰边界不动")
ax.set_xlabel("列 c"); ax.set_ylabel("行 r")
ax.grid(True, color="gray", lw=0.5)

# 右：世界模型框架
ax = axes[1]
ax.axis("off")
flow = [
    ("当前状态 s_t（图像 token）", "ViT 把摄像头画面变 token"),
    ("动作 a_t（马达指令）", "上下左右等"),
    ("世界模型 f(s_t, a_t)", "Transformer 预测"),
    ("下一状态 s_{t+1}（预测画面）", "「我这么做世界会怎样」"),
    ("→ 机器人据此规划", "挑能到达目标的那一步"),
]
for i, (step_txt, note) in enumerate(flow):
    y = 0.92 - i * 0.185
    ax.text(0.05, y, step_txt, fontsize=11, weight="bold", color="navy")
    ax.text(0.05, y - 0.055, note, fontsize=9, color="gray")
    if i < len(flow) - 1:
        ax.annotate("", xy=(0.05, y - 0.09), xytext=(0.05, y - 0.135),
                    arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_title("世界模型：学会「想象未来」", fontsize=13)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("可视化图/Day45_世界模型.png", dpi=120)
plt.show()
