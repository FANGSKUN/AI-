# Day 27：PnP（Perspective-n-Point）——已知 3D 点 + 2D 投影，反求相机位姿（R, t）
# 这是 Day 23/26「三角化（2D→3D）」的镜像问题：PnP 是「3D→位姿」
# 用途：SLAM 重定位、视觉里程计、AR 追踪——"我知道这些 3D 点长啥样，求我现在在哪朝哪看"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ================= 设定：内参 + 真实位姿（这就是你要反求的"答案"）=================
fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

C_true = np.array([1.5, 0.8, 2.0])   # 相机光心（世界坐标）
target = np.array([0.0, 0.0, 0.0])   # 相机看向原点

def look_at(C, target):
    # 复用 Day 26：让相机 z 轴指向 target，返回 R（第 i 行 = 相机第 i 轴在世界下的方向）
    z = target - C
    z = z / np.linalg.norm(z)
    up = np.array([0, 0, 1])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)

R_true = look_at(C_true, target)
t_true = -R_true @ C_true   # 约定 Xc = R(Xw - C) = R·Xw + t，所以 t = -R·C

# ================= 第 1 步：造 3D 点（世界坐标）=================
def make_points():
    # 立方体 8 个顶点（边长 1，中心在原点，顶点 = ±0.5 三种组合）+ 中心点 (0,0,0)
    idx = [-0.5, 0.5]
    pts = np.zeros((9, 3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pts[4*i + 2*j + k] = [idx[i], idx[j], idx[k]]
    pts[8] = [0, 0, 0]
    # 返回 (N, 3)
    return pts

# ================= 第 2 步：投影（正推，制造 2D 观测数据）=================
def project(Xw, K, R, C):
    # Xw (N,3) → 像素坐标 (N,2)
    Xc = R @ (Xw - C).T      # (3, N)
    X = K @ Xc               # (3, N)：行=相机轴(x,y,z)，列=点
    s = X[2]                 # 第 2 行 = 深度 z
    u = X[0] / s             # 第 0 行 = x，除以深度
    v = X[1] / s             # 第 1 行 = y，除以深度
    return np.stack([u, v], axis=1)   # (N, 2)

# ================= 第 3 步：归一化（像素 → x̂ = K⁻¹ x）=================
def normalize(uv, K):
    # uv (N,2) 像素坐标 → 归一化坐标 (N,2)，第三维隐含为 1
    N = uv.shape[0]
    xh = np.hstack([uv, np.ones((N, 1))])   # (N,3) 齐次像素坐标
    xn = (np.linalg.inv(K) @ xh.T).T        # (N,3) 归一化坐标
    return xn[:, :2]                        # (N,2)

# ================= 第 4 步：build_A（每个点贡献 2 个方程）=================
def build_A(xn, Xw):
    # TODO：xn (N,2) 归一化坐标，Xw (N,3) → A (2N, 12)
    # 核心：x̂ × (P X̃) = 0（投影方向 x̂ 与 P X̃ 平行 → 叉积为零）
    # 把 P 拉平成 12 维向量 p = [p1, p2, p3]（每行 4 个元素），每个点贡献两行：
    #   y·(p3 X̃) - (p2 X̃) = 0  →  [0,0,0,0,  -X,-Y,-Z,-1,   yX,yY,yZ,y] · p = 0
    #   (p1 X̃) - x·(p3 X̃) = 0  →  [X,Y,Z,1,   0,0,0,0,    -xX,-xY,-xZ,-x] · p = 0
    N = Xw.shape[0]
    A = np.zeros((2*N, 12))
    for i in range(N):
        x, y = xn[i]
        X, Y, Z = Xw[i]
        A[2*i]   = [0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y]
        A[2*i+1] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
    return A

# ================= 第 5 步：solve_P（SVD 求零空间）=================
def solve_P(A):
    # TODO：A (2N,12) → P (3,4)
    _,_,Vt=np.linalg.svd(A)
    e = Vt[-1]
    E= e.reshape(3,4)
    return E
    # 提示：和八点法一样，Vt[-1] 是 A p=0 的零空间解，reshape(3,4)

# ================= 第 6 步：decompose_P（从 P 拆出 R、t，处理尺度）=================
def decompose_P(P):
    # P (3,4) → R (3,3), t (3,)
    M = P[:, :3]                      # 前 3 列 = 尺度 μ × 旋转 R
    t = P[:, 3]                       # 第 4 列 = μ × t
    U, S, Vt = np.linalg.svd(M)       #这里的S充当一个放缩矩阵，省略了S就是把尺度恢复回去
    R = U @ Vt                        # 正交化，得到最接近 M 的旋转矩阵
    if np.linalg.det(R) < 0:          # 修正 det（负尺度 → 反射）
        R = -R
        mu = -S.mean()
    else:
        mu = S.mean()                 # 尺度 = 奇异值均值（M = μR，奇异值全 μ）
    t = t / mu                        # 除尺度还原真 t
    return R, t

# ================= 第 7 步：验证 + 可视化 ==================
def main():
    Xw = make_points()
    uv = project(Xw, K, R_true, C_true)
    xn = normalize(uv, K)
    A = build_A(xn, Xw)
    P = solve_P(A)
    R_est, t_est = decompose_P(P)

    # 反解相机光心：t = -R·C → C = -Rᵀ·t
    C_est = -R_est.T @ t_est

    print("=== PnP 反解位姿 vs 真实位姿 ===")
    print("R 误差（Frobenius）:", round(np.linalg.norm(R_est - R_true), 6))
    print("t 误差:", round(np.linalg.norm(t_est - t_true), 6))
    print("光心 C 误差:", round(np.linalg.norm(C_est - C_true), 6))
    print("det(R_est) =", round(np.linalg.det(R_est), 6), "（应为 1）")

    # 重投影误差（用反解的 R,t 重新投影，和真实 uv 对比，越小越好）
    uv_est = project(Xw, K, R_est, C_est)
    reproj_err = np.sqrt(((uv - uv_est) ** 2).sum(axis=1)).mean()
    print("平均重投影误差（像素）:", round(reproj_err, 6))

    # 3D 可视化：立方体 + 真实相机 vs 反解相机
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Xw[:, 0], Xw[:, 1], Xw[:, 2], c="royalblue", s=40, label="3D 点（立方体）")
    ax.scatter(*C_true, c="limegreen", s=120, marker="o", label="真实相机 C_true")
    ax.scatter(*C_est, c="crimson", s=60, marker="*", label="反解相机 C_est")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    ax.set_title("PnP：反解相机位姿 vs 真实位姿")
    plt.savefig("可视化图/Day27_PnP.png", dpi=150, bbox_inches="tight")
    print("已保存 可视化图/Day27_PnP.png")

if __name__ == "__main__":
    main()
