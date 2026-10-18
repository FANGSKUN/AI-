# -*- coding: utf-8 -*-
r"""
Day11 项目：正则化 + 早停——怎么"防"过拟合
===========================================
Day 10 我们亲眼看到：大模型(hidden=64)训练 1.0、验证 0.89，gap +0.11（过拟合）。
今天学两个"解药"，看它们怎么把这个 gap 压下去：

  1. L2 正则化（weight decay）：
     在 loss 里加一项"权重的平方和 × λ"。权重越大，罚得越狠。
     大权重 = 边界为了绕点而"猛拐"；惩罚它 → 边界更平滑 → 泛化更好。

  2. 早停（early stopping）：
     训练时盯着验证集 loss，一旦它连续好多轮不降反升，就提前停 + 回退到历史最优，
     别让它"背过头"。

把标了 TODO 的地方补全，然后运行 Day11_自检.py 检查。

运行：
    conda activate ai
    python Day11_自检.py
"""
import copy

import torch
import torch.nn as nn


# ============ 1. 复用 Day10 的数据 / 划分 / 评估 ============

def make_data(n=400, seed=0, noise=1.0):
    torch.manual_seed(seed)
    n0 = n // 2
    X0 = torch.randn(n0, 2) * noise + torch.tensor([-1.0, -1.0])
    X1 = torch.randn(n - n0, 2) * noise + torch.tensor([1.0, 1.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n0, 1), torch.ones(n - n0, 1)], dim=0)
    return X, y


def split(X, y, train_ratio=0.25, seed=7):
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n)
    k = int(n * train_ratio)
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


def accuracy(model, X, y):
    with torch.no_grad():
        p = torch.sigmoid(model(X))
        pred = (p >= 0.5).float()
        return (pred == y).float().mean().item()


# ============ 2. 训练函数（这次多了两个"防过拟合"开关）============

def train_mlp(X_tr, y_tr, X_va, y_va, hidden=64, epochs=5000, lr=0.01,
              weight_decay=0.0, early_stop=False, patience=500, report_every=100):
    """训练 MLP，可选择是否用 weight_decay / early_stop。

    返回 (model, train_accs, val_accs, stopped_epoch)：
      - model          训练好的模型（若早停触发，是历史最优参数）
      - train_accs     训练集准确率曲线
      - val_accs       验证集准确率曲线
      - stopped_epoch  实际停在第几轮（没早停就是 epochs）
    """
    model = nn.Sequential(
        nn.Linear(2, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    loss_fn = nn.BCELoss()

    # TODO 1：给 optimizer 加上 weight_decay 参数（L2 正则化就靠它）
    #   L2 正则化不用你手写 λ·Σw²，PyTorch 直接在 optimizer 上传 weight_decay 就行：
    #       optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #   左边 weight_decay= 是 PyTorch 约定的参数名；
    #   右边 weight_decay 是上面函数形参（默认 0.0 = 不惩罚，越大惩罚越狠）。
    optimizer = torch.optim.Adam(model.parameters(), lr=lr ,weight_decay= weight_decay)

    train_accs, val_accs = [], []
    best_val_loss = float("inf")   # 历史最优验证 loss（初始设为无穷大）
    best_state = None              # 历史最优时的参数副本
    no_improve = 0                 # 连续多少轮没改善
    stopped_epoch = epochs

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(torch.sigmoid(model(X_tr)), y_tr)
        loss.backward()
        optimizer.step()

        if epoch % report_every == 0:
            train_accs.append(accuracy(model, X_tr, y_tr))
            val_accs.append(accuracy(model, X_va, y_va))

        # TODO 2：早停逻辑（每次迭代都检查验证 loss）
        if early_stop:
            with torch.no_grad():
                val_loss = loss_fn(torch.sigmoid(model(X_va)), y_va).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                else:
                    no_improve += 1
                    if no_improve == patience:
                        stopped_epoch =epoch
                        break
            # (a) 若 val_loss 比 best_val_loss 还小：
            #       更新 best_val_loss、用 copy.deepcopy 存下 model.state_dict()、no_improve 清零
            # (b) 否则 no_improve += 1；当 no_improve >= patience 时：
            #       记下 stopped_epoch = epoch，然后 break
            # 下面写你的实现

    # 早停触发后，回退到历史最优参数
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)

    return model, train_accs, val_accs, stopped_epoch


# ============ 3. 主程序：三种方法对比 ============

def main():
    X, y = make_data()
    X_tr, y_tr, X_va, y_va = split(X, y)

    # 基线（Day10 的大模型，啥都不加）
    base, b_tr, b_va, _ = train_mlp(X_tr, y_tr, X_va, y_va, hidden=64)
    # 加 weight_decay
    wd, w_tr, w_va, _ = train_mlp(X_tr, y_tr, X_va, y_va, hidden=64, weight_decay=0.01)
    # 加 early_stop
    es, e_tr, e_va, ep = train_mlp(X_tr, y_tr, X_va, y_va, hidden=64,
                                   early_stop=True, patience=300)

    print("=" * 56)
    print(f"{'方法':<18}{'训练':>8}{'验证':>8}{'gap':>8}")
    print(f"{'基线(无正则)':<16}{b_tr[-1]:>8.3f}{b_va[-1]:>8.3f}{b_tr[-1] - b_va[-1]:>+8.3f}")
    print(f"{'weight_decay':<16}{w_tr[-1]:>8.3f}{w_va[-1]:>8.3f}{w_tr[-1] - w_va[-1]:>+8.3f}")
    print(f"{'early_stop':<16}{e_tr[-1]:>8.3f}{e_va[-1]:>8.3f}{e_tr[-1] - e_va[-1]:>+8.3f}")
    print(f"\nearly_stop 在第 {ep} 轮停下（共 5000 轮）")
    print("gap 越小 = 过拟合越轻")
    print("=" * 56)


if __name__ == "__main__":
    main()
