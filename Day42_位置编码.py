# Day 42：位置编码（Positional Encoding）——给注意力装上「顺序感」
# 问题：Self-Attention 只看内容、无视顺序——「猫追猫」里两个「猫」谁是主语谁是宾语，它分不清
# 解决：给每个 token 加上「它在第几个位置」的编码 PE，加到 embedding 上：X' = X + PE
# 论文用正弦余弦：PE[pos,2i]=sin(pos/10000^(2i/d))，PE[pos,2i+1]=cos(pos/10000^(2i/d))

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 场景：句子「猫 追 猫」，两个「猫」embedding 完全相同 =================
# 第一个「猫」是主语（位置 0），第二个「猫」是宾语（位置 2），但内容一模一样
d = 4                        # embedding 维度
seq_len = 3                  # 3 个 token（3 个位置）
X = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],    # X[0]: (4,) 位置0「猫」（主语）
    [0.0, 1.0, 0.0, 0.0],    # X[1]: (4,) 位置1「追」
    [1.0, 0.0, 0.0, 0.0],    # X[2]: (4,) 位置2「猫」（宾语，和位置0 完全一样）
])                           # X: (3,4)

# ================= 先看问题：Self-Attention 无视顺序 =================
scores = X @ X.T            # scores: (3,3) 两两内积，只依赖内容
print("两个「猫」的 embedding 完全相同，Self-Attention 分数矩阵：")
print(scores.numpy())
print("→ 第 0 行和第 2 行完全一样：两个「猫」的关注一模一样，分不清谁是主语谁是宾语\n")

# ================= TODO 1：正弦余弦位置编码 =================
def positional_encoding(seq_len, d):
    """生成 (seq_len, d) 的位置编码矩阵
    偶数维 i（0,2,4..）用 sin(pos / 10000^(i/d))
    奇数维 i（1,3,5..）用 cos(pos / 10000^((i-1)/d))
    提示：对每个位置 pos 和每个维度 i，按奇偶填 sin 或 cos
    """
    pe = torch.zeros(seq_len, d)          # pe: (seq_len, d)
    # ---- 在这里填：循环 pos 和 i，按奇偶填 sin/cos ----
    for pos in range(seq_len):
        for i in range(d):
            if i%2==0:
                pe[pos,i]=np.sin(pos/10000**(i/d))
            else:
                pe[pos,i]=np.cos(pos/10000**((i-1)/d))
    # -------------------
    return pe

PE = positional_encoding(seq_len, d)      # PE: (3,4)

# ================= TODO 2：把位置编码加到 embedding 上 =================
# X' = X + PE（位置信息「叠加」到内容信息上）
X_pe = X+PE  # ---- 在这里填：X_pe = ? ----

# ================= TODO 3：验证位置编码生效 =================
# 加了 PE 后，两个「猫」（位置0 和位置2）的表示应该不同了
# 打印 X_pe[0]（主语猫）和 X_pe[2]（宾语猫），看它们是否区分开
# ---- 在这里填：打印对比 ----
print(X_pe[0],X_pe[2])
# ================= 可视化：位置编码热力图 =================
print("\n位置编码矩阵 PE（每行 = 一个位置的编码向量）：")
print(np.round(PE.numpy(), 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 左：位置编码热力图（sin/cos 条纹）
im = axes[0].imshow(PE.numpy(), cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
axes[0].set_xticks(range(d)); axes[0].set_xticklabels([f"维{i}" for i in range(d)])
axes[0].set_yticks(range(seq_len)); axes[0].set_yticklabels([f"位置{i}" for i in range(seq_len)])
axes[0].set_xlabel("embedding 维度")
axes[0].set_ylabel("token 位置")
axes[0].set_title("位置编码 PE：不同位置有不同「条纹」\n（sin/cos，红=+1 蓝=-1）")
plt.colorbar(im, ax=axes[0])

# 右：对比两个「猫」加 PE 前后的表示
ax = axes[1]
ax.axis("off")
ax.text(0.5, 0.95, "两个「猫」的表示", ha="center", fontsize=13, weight="bold")
ax.text(0.5, 0.82, f"加 PE 前（相同）:\n  X[0]=X[2]={X[0].numpy()}", ha="center", fontsize=10)
ax.text(0.5, 0.55, f"加 PE 后（不同）:\n  主语猫={np.round(X_pe[0].numpy(), 2)}\n  宾语猫={np.round(X_pe[2].numpy(), 2)}",
        ha="center", fontsize=10)
ax.text(0.5, 0.25, "位置编码让「同一个词\n在不同位置」有了不同身份", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.savefig("可视化图/Day42_位置编码.png", dpi=120)
plt.show()
