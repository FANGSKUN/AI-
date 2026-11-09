# Day 44：ViT（Vision Transformer）—— 把图像切成 patch 当「词」
# 核心思想（An Image is Worth 16x16 Words）：
#   文字 Transformer 把「词」当 token；ViT 把「图像的小方块 patch」当 token
#   流程：图像 → 切 patch → 展平 → 线性投影成 embedding → 加 [CLS] + 位置编码 → Transformer 块
#   [CLS] token 是「汇总员」：放在最前，其他 patch 的信息通过注意力流向它，最后用它做分类

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ================= 1. 构造一张 8×8 灰度图（画一个 4×4 方块，后面看 patch 切分）=================
image = torch.zeros(8, 8)          # image: (8,8) 单通道灰度
image[2:6, 2:6] = 1.0              # 中间 4×4 方块 = 1（亮区）

patch_size = 4                     # 每个 patch 4×4
d = 8                              # embedding 维度（patch 投影后的维度）

# ================= TODO 1：切 patch + 展平 =================
# 8×8 图切成 2×2 个 4×4 的 patch，再展平成 4 个 16 维向量
# 提示：image.unfold(0, 4, 4) 在维度0 用「大小4、步长4」滑窗 → (2, 8, 4)
#       再 .unfold(1, 4, 4) 在维度1 同样滑窗 → (2, 2, 4, 4)
#       最后 .contiguous().view(4, 16) 展平成 4 个 patch、每个 16 维
patches = image.unfold(0,4,4).unfold(1,4,4)  # ---- 在这里填：patches = ?（形状 (4,16)）----
patches = patches.contiguous().view(4,16)

# ================= TODO 2：线性投影 + [CLS] + 位置编码 =================
# 16 维 patch 线性投影成 8 维 embedding，加一个 [CLS] token，再加位置编码
W_proj = torch.randn(16, d) * 0.5       # W_proj: (16,8) 投影矩阵（patch 16维 → 8维）
cls_token = torch.zeros(1, d)           # cls_token: (1,8) [CLS] 汇总员（初始化为 0，真实 ViT 可学习）
# ---- 在这里填 ----
patch_emb = patches @ W_proj    # patch_emb = patches @ W_proj，形状 (4,8)
X = torch.cat([cls_token, patch_emb],dim = 0)            # X = cat([cls_token, patch_emb], dim=0)，形状 (5,8)
# ---- 位置编码（复用 Day 42 的 sin/cos，这里 5 个位置：1 个 [CLS] + 4 个 patch）----
def positional_encoding(seq_len, d):
    pe = torch.zeros(seq_len, d)
    for pos in range(seq_len):
        for i in range(d):
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos / (10000 ** (i / d)))
            else:
                pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d)))
    return pe
PE = positional_encoding(5, d)          # PE: (5,8)
X = X + PE                               # X: (5,8) 加位置编码

# ================= TODO 3：送进 Transformer 块，取 [CLS] 输出 =================
# 复用 Day 43 的完整块：注意力（残差+LN）→ FFN（残差+LN）
def layer_norm(x, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True)
    return (x - mu) / (sigma + eps)

def attention(X_in):
    scores = X_in @ X_in.T / np.sqrt(d)   # (5,5)
    weights = torch.softmax(scores, dim=-1)
    return weights @ X_in                 # (5,8)

def ffn(x, W1, b1, W2, b2):
    return torch.relu(x @ W1 + b1) @ W2 + b2

d_ff = 16
W1 = torch.randn(d, d_ff) * 0.5; b1 = torch.zeros(d_ff)
W2 = torch.randn(d_ff, d) * 0.5; b2 = torch.zeros(d)

# 一个完整 Transformer 块
X1 = layer_norm(X + attention(X))            # ① 注意力子层
X2 = layer_norm(X1 + ffn(X1, W1, b1, W2, b2))  # ② FFN 子层

cls_out = X2[0]   # ---- 在这里填：cls_out = X2[?]（取 [CLS] 那一行作为整张图的表示）----

# ================= 打印 =================
print("原始图像 8×8：")
print(image.numpy().astype(int))
print("\n切成的 4 个 patch（每个 4×4 展平成 16 维，前 16 个值）：")
print(np.round(patches.numpy(), 2))
print("\npatch 投影后 patch_emb（4×8）：")
print(np.round(patch_emb.numpy(), 3))
print("\n[CLS] + 4 个 patch 的完整 token 序列 X（5×8，第 0 行是 [CLS]）：")
print(np.round(X.numpy(), 3))
print("\n经过 Transformer 块后，[CLS] 输出（整张图的表示，8 维）：")
print(np.round(cls_out.numpy(), 3))

# ================= 可视化 =================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 左：原始图像 + patch 切分线
ax = axes[0]
ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
for i in range(1, 2):
    ax.axhline(i * patch_size - 0.5, color="red", lw=2)
    ax.axvline(i * patch_size - 0.5, color="red", lw=2)
ax.set_title("8×8 图像（红线 = patch 边界）\n切成 2×2 个 4×4 的 patch")
ax.set_xticks([]); ax.set_yticks([])

# 中：4 个 patch 各自内容
ax = axes[1]
for idx in range(4):
    r, c = divmod(idx, 2)
    ax.add_patch(plt.Rectangle((c * 1.1, (1 - r) * 1.1), 1, 1, fill=True, alpha=1,
                               facecolor="white" if patches[idx, 0] == 0 else "black",
                               edgecolor="red", lw=2))
    ax.text(c * 1.1 + 0.5, (1 - r) * 1.1 + 0.5, f"patch{idx}", ha="center", va="center",
            color="red", fontsize=11, weight="bold")
ax.set_xlim(0, 2.2); ax.set_ylim(0, 2.2)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("4 个 patch（黑=亮区/白=暗区）\n每个展平成 16 维向量")

# 右：ViT 流程
ax = axes[2]
ax.axis("off")
flow = [
    ("图像 8×8", "输入"),
    ("切 4 个 patch（4×4）", "把图拆成「词」"),
    ("展平 + 线性投影", "16 维 → 8 维 embedding"),
    ("[CLS] + 位置编码", "5 个 token 序列"),
    ("Transformer 块", "注意力让信息流动"),
    ("[CLS] 输出 → 分类", "整张图的表示"),
]
for i, (step, note) in enumerate(flow):
    y = 0.92 - i * 0.155
    ax.text(0.05, y, step, fontsize=11, weight="bold", color="navy")
    ax.text(0.05, y - 0.05, note, fontsize=9, color="gray")
    if i < len(flow) - 1:
        ax.annotate("", xy=(0.05, y - 0.08), xytext=(0.05, y - 0.12),
                    arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_title("ViT：图像 → patch → token → 分类", fontsize=13)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("可视化图/Day44_ViT.png", dpi=120)
plt.show()
