# Day 41：Multi-Head Attention（多头注意力）
# 单头只能学一种「关注模式」（一套 QKV 投影），多头 = 多个头并行，各用不同投影捕捉不同关系
# 公式：MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_o，其中 head_i = Attention(QW_q^i, KW_k^i, VW_v^i)

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 输入：4 个 token 的 embedding（3 维）=================
X = torch.tensor([
    [1.0, 0.0, 0.0],   # X[0]: (3,)  token0「猫」
    [0.0, 1.0, 0.0],   # X[1]: (3,)  token1「鱼」
    [1.0, 0.0, 0.0],   # X[2]: (3,)  token2「猫」
    [0.0, 0.0, 1.0],   # X[3]: (3,)  token3「鸟」
])                     # X: (4,3)

d = 3

# ================= 两个头的投影矩阵（各自独立）=================
# 头 1：W_k 置换（猫鸟标签互换）→ 捕捉「角色对调」关系
W_q1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # W_q1: (3,3)
W_k1 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # W_k1: (3,3) 置换
W_v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # W_v1: (3,3)

# 头 2：W_k 单位（标签不变）→ 捕捉「长得像」关系
W_q2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # W_q2: (3,3)
W_k2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # W_k2: (3,3) 单位
W_v2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # W_v2: (3,3)

# 输出投影：把拼接后的 6 维（两个头各 3 维）融合回 3 维（这里 = 两个头平均）
W_o = torch.tensor([
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],   # 头1 的三维，权重 1
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],   # 头2 的三维，权重 1
]) / 2.0                                                   # W_o: (6,3) 平均两个头

# ================= TODO 1：单头 attention 函数（封装 Day 40 的三步）=================
def attention(Q, K, V):
    """输入 Q,K,V（都是 (4,3)）→ 输出 (4,3)
    三步：scores = Q@K^T/√d → softmax → weights@V
    返回：output 和 weights（weights 用于可视化）
    """
    # ---- 在这里填 ----
    scores = Q@K.T/np.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    output = weights@V
    # -------------------
    return output, weights

# ================= TODO 2：两个头各自投影 + 算注意力 =================
# 头1 用 W_q1/W_k1/W_v1，头2 用 W_q2/W_k2/W_v2
head1, w1 = attention(X @ W_q1, X @ W_k1, X @ W_v1)  # ---- 在这里填：head1, w1 = attention(?, ?, ?) ----
head2, w2 = attention(X @ W_q2, X @ W_k2, X @ W_v2)  # ---- 在这里填：head2, w2 = attention(?, ?, ?) ----

# ================= TODO 3：拼接 + 输出投影 =================
# 把 head1、head2 沿最后一维拼接 → (4,6)，再 @ W_o → (4,3)
multi = torch.cat([head1, head2], dim=-1)   # ---- 在这里填：multi = ?（拼接）----
output = multi@W_o  # ---- 在这里填：output = ?（线性变换）----

# ================= 打印结果 =================
print("头1 的注意力权重（猫鸟角色对调关系）：")
print(np.round(w1.numpy(), 3))
print("\n头2 的注意力权重（长得像关系）：")
print(np.round(w2.numpy(), 3))
print("\n拼接后 multi（4×6，前3维=头1、后3维=头2）：")
print(np.round(multi.numpy(), 3))
print("\n多头最终输出 output（= 两个头的平均）：")
print(np.round(output.numpy(), 3))

# ================= 可视化：两个头的注意力对比 =================
words = ["猫", "鱼", "猫", "鸟"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, w, title in [
    (axes[0], w1, "头1：猫鸟角色对调\n（猫关注鸟）"),
    (axes[1], w2, "头2：长得像\n（猫关注猫）"),
]:
    im = ax.imshow(w.numpy(), cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{w[i, j].item():.2f}", ha="center", va="center",
                    color="white" if w[i, j].item() > 0.5 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(words)
    ax.set_yticks(range(4)); ax.set_yticklabels(words)
    ax.set_xlabel("被关注的 token（Key）")
    ax.set_ylabel("当前 token（Query）")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

# 右图：多头最终输出，标出每个 token 融合了哪两个头的信息
ax = axes[2]
ax.axis("off")
ax.text(0.5, 0.9, "多头最终输出", ha="center", fontsize=13, weight="bold")
for i in range(4):
    ax.text(0.5, 0.7 - i * 0.15, f"{words[i]}: {np.round(output[i].numpy(), 3)}",
            ha="center", fontsize=11)
ax.text(0.5, 0.05, "= 头1 和 头2 的平均\n（两种关系各贡献一半）", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.savefig("可视化图/Day41_多头注意力.png", dpi=120)
plt.show()
