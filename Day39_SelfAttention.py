# Day 39：Transformer 入门——Self-Attention（自注意力）
# 核心思想：每个 token 的输出 = 根据「相似度」对所有 token 的加权平均
#   公式：Attention(Q,K,V) = softmax(Q·K^T / √d) · V
#   今天用最简情况 Q=K=V=X（每个 token 拿自己查自己），先看清「注意力」本身

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 场景：4 个 token 的 embedding（3 维）=================
# token0 和 token2 是同一个词「猫」，embedding 完全相同；其余各不相同
X = torch.tensor([
    [1.0, 0.0, 0.0],   # X[0]: (3,)  token0「猫」
    [0.0, 1.0, 0.0],   # X[1]: (3,)  token1「鱼」
    [1.0, 0.0, 0.0],   # X[2]: (3,)  token2「猫」（和 token0 一样）
    [0.0, 0.0, 1.0],   # X[3]: (3,)  token3「鸟」
])                     # X: (4,3) 4 个 token，每个 3 维

d = X.shape[1]         # d = 3（embedding 维度）

# ================= TODO 1：注意力分数（相似度）=================
# 每个 token 和每个 token 算内积（点积 = 相似度）：scores[i,j] = token_i 和 token_j 的相似度
# 提示：X @ X.T 就是「两两内积」，结果形状 (4,4)
scores = X@X.T  # ---- 在这里填：scores = ? ----

# ================= TODO 2：缩放 + softmax =================
# ① 除以 √d（防止内积太大 → softmax 梯度消失）
# ② 对每一行（dim=-1）做 softmax → 权重归一化（每行和为 1）
weights = torch.softmax(scores/np.sqrt(d), dim=-1)  # ---- 在这里填：weights = ? ----

# ================= TODO 3：加权求和 =================
# 每个 token 的输出 = 所有 token 按注意力权重的加权平均
# output[i] = Σ_j weights[i,j] * V[j]（今天 V=X）
output = output = weights @ X   # ---- 在这里填：output = ? ----

# ================= 打印结果 =================
print("输入 X（4 个 token 的 embedding）：")
print(X.numpy())
print("\n注意力分数 scores = X @ X^T（两两内积，越大越相似）：")
print(scores.numpy())
print("\n注意力权重 weights = softmax(scores/√d)（每行归一化，和为 1）：")
print(np.round(weights.numpy(), 3))
print("\n输出 output = weights @ X（每个 token 聚合后的新表示）：")
print(np.round(output.numpy(), 3))

# ================= 可视化：注意力权重热力图 =================
words = ["猫", "鱼", "猫", "鸟"]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(weights.numpy(), cmap="Blues")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{weights[i, j].item():.2f}", ha="center", va="center",
                color="white" if weights[i, j].item() > 0.5 else "black")
ax.set_xticks(range(4)); ax.set_xticklabels(words)
ax.set_yticks(range(4)); ax.set_yticklabels(words)
ax.set_xlabel("被关注的 token（Key）")
ax.set_ylabel("当前 token（Query）")
ax.set_title("注意力权重矩阵：每个 token 在「看」谁")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("可视化图/Day39_SelfAttention.png", dpi=120)
plt.show()
