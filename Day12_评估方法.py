# -*- coding: utf-8 -*-
r"""
Day12 项目：test 集 + 交叉验证——怎么"可靠地"评估模型
=====================================================
Day 10-11 我们一直用 train/val 两划分：train 训练、val 调超参数。

但这里藏着一个隐患：val 被你反复"偷看"来调参，调久了会"无意中对 val
也过拟合"——这叫**选择偏差**（selection bias）。val 选出的"最优"，可能
只是"在 val 上运气最好"的那个，未必真最优。

所以标准做法是**三划分** + **交叉验证**：

  1. train/val/test 三划分：
       train 训练参数 → val 调超参数/选模型 → test 最终只测一次、报告真实成绩。
       test 就像"高考"：只考一次，考完才知道真实水平；绝不能拿来调参。

  2. k 折交叉验证（k-fold CV）：
       数据少时，固定切一次 train/val 可能"切得巧/不巧"，结果不稳定。
       把数据切成 k 折，轮流拿 1 折当验证、剩下 k-1 折训练，训练 k 次取平均
       → 每个样本都当过验证，评估更稳、数据更充分。

把标了 TODO 的地方补全，然后运行 Day12_自检.py 检查。

运行：
    conda activate ai
    python Day12_自检.py
"""
import torch
import torch.nn as nn


# ============ 1. 复用数据 / 评估 ============

def make_data(n=400, seed=0, noise=1.0):
    torch.manual_seed(seed)
    n0 = n // 2
    X0 = torch.randn(n0, 2) * noise + torch.tensor([-1.0, -1.0])
    X1 = torch.randn(n - n0, 2) * noise + torch.tensor([1.0, 1.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n0, 1), torch.ones(n - n0, 1)], dim=0)
    return X, y


def accuracy(model, X, y):
    with torch.no_grad():
        p = torch.sigmoid(model(X))
        pred = (p >= 0.5).float()
        return (pred == y).float().mean().item()


# ============ 2. 训练函数（简单 MLP；今天重点在"评估"，不在"训练"）============

def train_model(X_tr, y_tr, hidden=16, epochs=2000, lr=0.01, seed=9):
    """训练一个固定的小 MLP，返回训练好的模型。"""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(torch.sigmoid(model(X_tr)), y_tr)
        loss.backward()
        opt.step()
    return model


# ============ 3. 今天的两个 TODO ============

def split_three(X, y, train_ratio=0.6, val_ratio=0.2, seed=7):
    """把数据切成 train / val / test 三块，返回 (X_tr, y_tr, X_va, y_va, X_te, y_te)。

    思路（照着填）：
      1. 先 torch.manual_seed(seed) + torch.randperm(n) 打乱（顺序切会全是一类！）
      2. 算 n_tr = int(n * train_ratio)，n_va = int(n * val_ratio)
      3. 用 idx 切片切三段：
           train = idx[:n_tr]
           val   = idx[n_tr : n_tr + n_va]
           test  = idx[n_tr + n_va :]
         X 和 y 要用同一套索引（X[idx] 与 y[idx] 一一对应）。
    """
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    train = idx[:n_tr]
    val = idx[n_tr : n_tr + n_va]
    test = idx[n_tr + n_va :]
    X_tr = X[train]
    y_tr = y[train]
    X_va = X[val]
    y_va = y[val]
    X_te = X[test]
    y_te = y[test]
    # TODO 1：按 idx 切出 train / val / test 三块，返回 6 个张量。
    #   提示：X_tr = X[idx[:n_tr]]，y_tr = y[idx[:n_tr]]，其余同理。
    #   完成后把下面的 return None 换成 return X_tr, y_tr, X_va, y_va, X_te, y_te
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def kfold_scores(X, y, k=5, hidden=16, seed=7):
    """k 折交叉验证：把数据分成 k 折，轮流用 1 折验证、其余 k-1 折训练。

    返回 k 个验证准确率组成的列表。
    思路（照着填）：
      1. 打乱后把索引等分成 k 段（每段大约 n // k 个）
      2. 第 i 折：验证 = 第 i 段索引，训练 = 其余所有索引拼起来
      3. 用 train_model 在训练段上训练，accuracy 在验证段上打分
      4. 把 k 个分数收进列表返回
    """
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n)
    fold = n // k
    scores = []
    for i in range(k):
        tr_idx=torch.cat([idx[: i*fold],idx[i*fold : (i+1)*fold]])
        va_idx=idx[i*fold : (i+1)*fold]
        model = train_model(X[tr_idx], y[tr_idx], hidden=hidden)
        scores.append(accuracy(model, X[va_idx], y[va_idx]))
    # TODO 2：实现 k 折循环。
    #   提示：
    #     - 第 i 折的验证索引：idx[i * fold : (i + 1) * fold]
    #     - 训练索引 = 除了验证之外的索引，可用 torch.cat 拼两段：
    #         torch.cat([idx[: i * fold], idx[(i + 1) * fold :]])
    #     - model = train_model(X[tr_idx], y[tr_idx], hidden=hidden)
    #     - scores.append(accuracy(model, X[va_idx], y[va_idx]))
    #   完成后把下面的 return None 换成 return scores
    return scores


# ============ 4. 主程序：三件事的演示 ============

def main():
    X, y = make_data(400, seed=0)

    # ---- 演示 1：三划分 ----
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_three(X, y)
    print("=" * 60)
    print("演示 1：train / val / test 三划分")
    print(f"  总数 {len(y)} -> train {len(y_tr)} / val {len(y_va)} / test {len(y_te)}")

    # ---- 演示 2：用 val 选超参数，用 test 报告（选择偏差）----
    # 注意：为了让你"看到"选择偏差，这里故意把 test 也算出来了。
    # 真实流程里 test 绝不能在这个阶段被偷看，只能最后测一次！
    print("=" * 60)
    print("演示 2：用 val 选 hidden，再看它在 test 上是不是真最优")
    print(f"{'hidden':>8}{'val':>10}{'test':>10}")
    rows = []
    for h in [4, 8, 16, 32, 64, 128]:
        m = train_model(X_tr, y_tr, hidden=h)
        va = accuracy(m, X_va, y_va)
        te = accuracy(m, X_te, y_te)   # 演示用，真实流程里不要这样
        rows.append((h, va, te))
        print(f"{h:>8}{va:>10.3f}{te:>10.3f}")
    best_val = max(rows, key=lambda r: r[1])
    best_te = max(rows, key=lambda r: r[2])
    print(f"  val 选出的最优 hidden={best_val[0]}（val {best_val[1]:.3f}）")
    print(f"  test 真正最优   hidden={best_te[0]}（test {best_te[2]:.3f}）")
    if best_val[0] != best_te[0]:
        print("  >>> 看到了吗：val 选出的 ≠ test 真正最优，这就是选择偏差。")
    else:
        print("  >>> 这次恰好一致（数据小、候选少），多跑几次 seed 会看到不一致。")

    # ---- 演示 3：固定切分 vs k 折（稳定性）----
    print("=" * 60)
    print("演示 3：固定切分(5 个不同 seed) vs k 折交叉验证")
    fixed = []
    for s in range(5):
        a_tr, a_ytr, a_va, a_yva, _, _ = split_three(X, y, seed=s)
        m = train_model(a_tr, a_ytr)
        fixed.append(accuracy(m, a_va, a_yva))
    kf = kfold_scores(X, y, k=5)
    import statistics
    fixed_str = ", ".join(f"{v:.3f}" for v in fixed)
    kf_str = ", ".join(f"{v:.3f}" for v in kf)
    print(f"  固定切分: [{fixed_str}]  均值 {statistics.mean(fixed):.3f} 标准差 {statistics.stdev(fixed):.3f}")
    print(f"  k 折CV  : [{kf_str}]  均值 {statistics.mean(kf):.3f} 标准差 {statistics.stdev(kf):.3f}")
    print("  标准差分更小 = 评估更稳定（少受「切得巧不巧」影响）")
    print("=" * 60)


if __name__ == "__main__":
    main()
