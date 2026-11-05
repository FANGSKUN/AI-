# Day 40：QKV 线性投影——Self-Attention 的完整形态
# 昨天 Q=K=V=X（拿自己查自己），今天 Q=XW_q、K=XW_k、V=XW_v（三个投影，可学习）
# 核心：投影让「什么算相关」从「向量长得像」升级成「语义上该不该关注」
#   Q（Query 查询）：我在找什么
#   K（Key 键）    ：我身上贴了什么标签
#   V（Value 值）  ：我实际装了什么内容

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
    [1.0, 0.0, 0.0],   # X[2]: (3,)  token2「猫」（和 token0 一样）
    [0.0, 0.0, 1.0],   # X[3]: (3,)  token3「鸟」
])                     # X: (4,3)

# ================= 三个投影矩阵（今天手造，真实情况是可学习的参数）=================
# 故事：W_k 是「置换矩阵」——它把「猫」的标签 [1,0,0] 换成 [0,0,1]，把「鸟」的标签换成 [1,0,0]
# 效果：投影后「猫」的查询，会匹配到「鸟」的标签 → 展示「投影重新定义谁和谁相关」
W_q = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])                     # W_q: (3,3) 单位阵：Q=X（查询保持原样）

W_k = torch.tensor([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
])                     # W_k: (3,3) 置换阵：K 空间里「猫」和「鸟」标签互换

W_v = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])                     # W_v: (3,3) 单位阵：V=X（内容保持原样）

# ================= TODO 1：三个投影 =================
# Q = X @ W_q、K = X @ W_k、V = X @ W_v（每个形状都还是 (4,3)）
Q = X@W_q  # ---- 在这里填 ----
K = X@W_k  # ---- 在这里填 ----
V = X@W_v  # ---- 在这里填 ----

d_k = K.shape[1]       # 投影后的维度（这里 = 3）

# ================= TODO 2：注意力（和 Day 39 一样，但用 Q、K）=================
# scores = Q @ K^T / √d_k；weights = softmax(scores, dim=-1)
scores =Q @ K.T / np.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)     # ---- 在这里填 ----

# ================= TODO 3：加权求和（用 V，不是 X）=================
output = weights@V  # ---- 在这里填 ----

# ================= 打印结果 =================
print("K = X @ W_k（「猫」的标签从 [1,0,0] 变成了 [0,0,1]）：")
print(K.numpy())
print("\n注意力分数 scores = Q @ K^T / √d：")
print(np.round(scores.numpy(), 3))
print("\n注意力权重 weights（每行 = 一个 token 的关注分布）：")
print(np.round(weights.numpy(), 3))
print("\n输出 output = weights @ V：")
print(np.round(output.numpy(), 3))

# ================= 可视化：对比 Day 39（无投影）vs Day 40（K 投影）=================
words = ["猫", "鱼", "猫", "鸟"]

# Day 39 的注意力（Q=K=V=X，无投影）
scores_old = X @ X.T
weights_old = torch.softmax(scores_old / np.sqrt(d_k), dim=-1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, w, title in [
    (axes[0], weights_old, "Day39：无投影（Q=K=X）\n猫关注猫（长得像）"),
    (axes[1], weights, "Day40：K 投影后\n猫关注鸟（标签被投影重新定义）"),
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
plt.tight_layout()
plt.savefig("可视化图/Day40_QKV投影.png", dpi=120)
plt.show()
