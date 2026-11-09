# Day 46：训练真·世界模型——删掉 step，让神经网络自己学规则
# 核心：step 函数只用来「造训练数据」（提供标准答案），训练时模型完全看不到它，
#       模型只能从「(当前帧, 动作) → 下一帧」的样本对里自己猜出规律 → 这才是预测
# step 的三个角色：①造数据(给答案) ②训练时=看不见 ③测试时=只用来对答案

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)
np.random.seed(0)

# ========== 世界规则（上帝视角，只用来造数据/对答案，模型永远看不见）==========
SIZE = 8
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # 0上 1下 2左 3右
ACTION_NAMES = ["上", "下", "左", "右"]

def step(pos, action):
    """真实世界规则：球按动作移动，碰边界不动（这就是模型要学的『天机』）"""
    r, c = pos
    dr, dc = ACTIONS[action]
    return (max(0, min(SIZE - 1, r + dr)), max(0, min(SIZE - 1, c + dc)))

def frame_to_token(pos):
    """位置 (r,c) → 64 维 one-hot token"""
    t = torch.zeros(SIZE, SIZE)
    t[pos[0], pos[1]] = 1.0
    return t.reshape(-1)                      # (64,)

def action_to_token(a):
    """动作 → 4 维 one-hot"""
    t = torch.zeros(4)
    t[a] = 1.0
    return t                                   # (4,)

# ========== TODO 1：生成训练数据 ==========
# 随机采样 n 个 (位置, 动作)，用 step 算出「下一位置」当标准答案
# X 每一行 = [当前帧(64) | 动作(4)] = 68 维；y = 下一帧的格编号(0~63)
def generate_data(n):
    xs, ys = [], []
    for _ in range(n):
        pos = (np.random.randint(SIZE), np.random.randint(SIZE))
        a = np.random.randint(4)
        next_pos = step(pos,a)          # ---- 在这里填：step(pos, a) ----
        x = torch.cat([frame_to_token(pos), action_to_token(a)])   # (68,)
        y = next_pos[0] * SIZE + next_pos[1]      # ---- 在这里填：把 next_pos 转成格编号 0~63（r*SIZE+c）----
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys)   # (n,68) 和 (n,)

# ========== TODO 2：世界模型网络（MLP）==========
# 输入 68 维（当前帧 + 动作），输出 64 维 logits（预测下一帧每格是球的分数）
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(68, 128),   # ---- 在这里填：输入维度（帧+动作的拼接长度）----
            nn.ReLU(),
            nn.Linear(128, 64),   # ---- 在这里填：输出维度（预测哪一格）----
        )

    def forward(self, x):
        return self.net(x)          # (n, 64) logits

# ========== 生成数据 & 初始化模型 ==========
X, y = generate_data(400)          # 400 个随机样本（总共 256 种组合，随机采 400 会有重复，正常）
model = WorldModel()
criterion = nn.CrossEntropyLoss()  # 交叉熵：预测「下一格」的分数 vs 真实格编号
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ========== TODO 3：训练循环 ==========
# 反复：前向 → 算损失 → 反向求梯度 → 更新参数
epochs = 500
loss_history = []
for epoch in range(epochs):
    logits = model(X)              # 前向：预测 (n,64)
    loss = criterion(logits, y)   # ---- 在这里填：criterion(logits, y) ----
    optimizer.zero_grad()          # 清空上一轮梯度
    loss.backward()                # 反向：算每个参数的梯度
    optimizer.step()               # 更新：沿梯度方向挪一小步
    loss_history.append(loss.item())

print(f"训练完成，最终 loss = {loss_history[-1]:.4f}")

# ========== 测试：模型对「训练样本之外」的局面预测得对不对 ==========
# 抽样一些 (位置, 动作)，用训练好的模型预测下一格，和真实 step 对答案
model.eval()
test_poss = [(0, 0), (7, 7), (3, 5), (0, 3), (7, 2)]
test_acts = [0, 1, 3, 2, 0]
print("\n模型预测 vs 真实世界（step 对答案）：")
all_correct = True
for pos, a in zip(test_poss, test_acts):
    x = torch.cat([frame_to_token(pos), action_to_token(a)]).unsqueeze(0)  # (1,68)
    logits = model(x)                    # (1,64)
    pred_idx = logits.argmax(dim=1).item()
    pred_pos = (pred_idx // SIZE, pred_idx % SIZE)
    true_pos = step(pos, a)
    ok = (pred_pos == true_pos)
    all_correct = all_correct and ok
    print(f"  球在{pos} 按「{ACTION_NAMES[a]}」→ 模型预测{pred_pos}，真实{true_pos}  {'✓' if ok else '✗'}")

print(f"\n5 个测试 {'全对，模型学会了规则！' if all_correct else '有错（见上）'}")

# ========== 可视化：训练曲线 + 模型学到的「世界」==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：loss 下降曲线
ax = axes[0]
ax.plot(loss_history, color="crimson", lw=1.5)
ax.set_title("训练 loss 下降（模型在一点点逼近规则）")
ax.set_xlabel("epoch"); ax.set_ylabel("交叉熵 loss")
ax.grid(True, alpha=0.3)

# 右：展示一个「预测轨迹」——模型自己脑补的未来 vs 真实
ax = axes[1]
ax.axis("off")
traj_pred = [(4, 4)]
traj_true = [(4, 4)]
pos_p, pos_t = (4, 4), (4, 4)
for a in [3, 1, 3, 3, 2, 0]:     # 右 下 右 右 左 上
    x = torch.cat([frame_to_token(pos_p), action_to_token(a)]).unsqueeze(0)
    idx = model(x).argmax(dim=1).item()
    pos_p = (idx // SIZE, idx % SIZE)
    pos_t = step(pos_t, a)
    traj_pred.append(pos_p)
    traj_true.append(pos_t)

txt = "「脑补」轨迹 vs 真实轨迹：\n\n"
for i, (p, t) in enumerate(zip(traj_pred, traj_true)):
    mark = "✓" if p == t else "✗"
    txt += f"  第{i}帧  模型{p}  真实{t}  {mark}\n"
ax.text(0.05, 0.5, txt, fontsize=11, va="center", family="monospace")
ax.set_title("模型学会自己「想象未来」了", fontsize=13)

plt.tight_layout()
plt.savefig("可视化图/Day46_训练世界模型.png", dpi=120)
plt.show()
