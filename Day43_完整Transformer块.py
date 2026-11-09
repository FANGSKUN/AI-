# Day 43：FFN（前馈 MLP）+ LayerNorm（层归一化）—— 完整 Transformer 块
# 完整块 = 注意力子层（残差+LayerNorm） + FFN 子层（残差+LayerNorm）
# 注意力负责「token 之间交换信息」（横向），FFN 负责「每个 token 自己加工」（纵向）
# LayerNorm 把每个 token 的特征向量归一化（均值 0、方差 1），稳定深层训练

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 场景：句子「猫 追 猫」，3 个 token，4 维 embedding =================
d = 4
seq_len = 3
X = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # X[0]: (4,) 位置0「猫」（主语）
    [0.0, 1.0, 0.0, 0.0],   # X[1]: (4,) 位置1「追」
    [1.0, 0.0, 0.0, 0.0],   # X[2]: (4,) 位置2「猫」（宾语）
])                          # X: (3,4)

# ================= 位置编码（复用 Day 42）=================
def positional_encoding(seq_len, d):
    pe = torch.zeros(seq_len, d)
    for pos in range(seq_len):
        for i in range(d):
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos / (10000 ** (i / d)))
            else:
                pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d)))
    return pe

PE = positional_encoding(seq_len, d)          # PE: (3,4)
X_pe = X + PE                                  # X_pe: (3,4) 位置编码后的 token

# ================= 单头注意力（Q=K=V=X_pe，最简，重点看 FFN + LayerNorm）=================
def attention(X_in):
    scores = X_in @ X_in.T / np.sqrt(d)        # scores: (3,3)
    weights = torch.softmax(scores, dim=-1)    # weights: (3,3)
    return weights @ X_in                      # output: (3,4)

Attn = attention(X_pe)                         # Attn: (3,4) 注意力输出

# ================= TODO 1：LayerNorm（对每个 token 的 4 维向量归一化）=================
def layer_norm(x, eps=1e-6):
    """对每个 token（每行）自己的 d 维向量归一化：均值 0、方差 1
    注意：是按「最后一维」算均值/方差，不是跨 token 算
    提示：x.mean(dim=-1, keepdim=True)、x.std(dim=-1, keepdim=True)
    """
    # ---- 在这里填 ----
    mu = x.mean(dim=-1, keepdim=True)  # (3,1) 每行的均值
    sigma = x.std(dim=-1, keepdim=True)      # (3,1) 每行的标准差
    out = (x - mu) / (sigma + eps)
    # -------------------
    return out

# ================= FFN 参数：4 维升到 8 维再降回 4 维 =================
d_ff = 8
W1 = torch.randn(d, d_ff) * 0.5   # W1: (4,8) 升维
b1 = torch.zeros(d_ff)            # b1: (8,)
W2 = torch.randn(d_ff, d) * 0.5   # W2: (8,4) 降维
b2 = torch.zeros(d)               # b2: (4,)

# ================= TODO 2：FFN（两层 MLP + ReLU，升维再降维）=================
def ffn(x, W1, b1, W2, b2):
    """FFN(x) = W2 @ ReLU(W1 @ x + b1) + b2
    先升维（d→d_ff，给更多容量），ReLU 引入非线性，再降维（d_ff→d）
    每个 token 独立处理（不和其他 token 交互）
    """
    # ---- 在这里填 ----
    hidden = torch.relu(x @ W1 + b1)
    out = hidden @ W2 + b2
    # -------------------
    return out

# ================= TODO 3：组装完整 Transformer 块 =================
# ① 注意力子层：残差 + LayerNorm
#    X1 = layer_norm(X_pe + Attn)
# ② FFN 子层：残差 + LayerNorm
#    F  = ffn(X1, W1, b1, W2, b2)
#    X2 = layer_norm(X1 + F)
# ---- 在这里填：两步组装 ----
X1 = layer_norm(X_pe + Attn)   # 注意力子层输出
F = ffn(X1, W1, b1, W2, b2)    # FFN 输出
X2 = layer_norm(X1 + F)   # 完整块最终输出
# -------------------

# ================= 打印结果 =================
raw = X_pe + Attn                              # 残差后、归一化前的值
print("注意力子层残差后（未归一化）raw = X_pe + Attn：")
print(np.round(raw.numpy(), 3))
print("\nLayerNorm 后 X1（每行均值 0、方差 1）：")
print(np.round(X1.numpy(), 3))
print("  每行均值：", np.round(X1.mean(dim=-1).numpy(), 4))
print("  每行方差：", np.round(X1.var(dim=-1).numpy(), 4))
print("\nFFN 输出 F（升维 4→8 加工后再降回 4）：")
print(np.round(F.numpy(), 3))
print("\n完整块最终输出 X2：")
print(np.round(X2.numpy(), 3))

# ================= 可视化 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：LayerNorm 前后对比（取主语猫那一行）
ax = axes[0]
row = 0   # 主语猫
dims = np.arange(d)
raw_row = raw[row].numpy()
norm_row = X1[row].numpy()
x_pos = np.arange(d)
ax.bar(x_pos - 0.2, raw_row, 0.4, label="归一化前 raw", color="orange")
ax.bar(x_pos + 0.2, norm_row, 0.4, label="归一化后 X1", color="steelblue")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels([f"维{i}" for i in range(d)])
ax.set_xlabel("特征维度")
ax.set_ylabel("数值")
ax.set_title("LayerNorm 把「猫」的 4 维数值\\n拉回均值 0、方差 1（数值被压缩到 ±1 附近）")
ax.legend()

# 右：完整 Transformer 块数据流
ax = axes[1]
ax.axis("off")
flow = [
    ("输入 X（3×4）", "猫 追 猫 的 embedding"),
    ("X_pe = X + PE", "加位置编码"),
    ("Attn = 注意力(X_pe)", "token 间交换信息（横向）"),
    ("X1 = LayerNorm(X_pe + Attn)", "残差 + 归一化 ①"),
    ("F = FFN(X1)", "每个 token 自己加工（纵向）"),
    ("X2 = LayerNorm(X1 + F)", "残差 + 归一化 ② = 块输出"),
]
for i, (step, note) in enumerate(flow):
    y = 0.92 - i * 0.16
    ax.text(0.08, y, step, fontsize=11, weight="bold", color="navy")
    ax.text(0.08, y - 0.055, note, fontsize=9, color="gray")
    if i < len(flow) - 1:
        ax.annotate("", xy=(0.08, y - 0.09), xytext=(0.08, y - 0.13),
                    arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_title("完整 Transformer 块（Encoder Block）", fontsize=13)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("可视化图/Day43_完整Transformer块.png", dpi=120)
plt.show()
