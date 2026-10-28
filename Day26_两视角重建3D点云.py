# ============================================================
# Day 26 · 项目实战：从两个视角重建 3D 点云（SfM 简化版）
# ============================================================
#
# 【目标】
#   造一个 3D 场景（N 个点），用两个相机位姿"拍照"得到两堆 2D 像素，
#   然后用八点法反解出相机运动（R、t），再用三角化把所有点重建回 3D，
#   最后可视化对比「真实点云」和「重建点云」。
#
#   换句话说：从两张 2D 照片，还原出一个 3D 点云 —— 这就是 SLAM 的核心。
#
# 【你将用到的知识】Day 21–25
#   - Day 21/22：内参 K、透视投影（3D → 2D 像素）
#   - Day 23    ：三角化（2D → 3D，反投影求交点）
#   - Day 25    ：八点法解 E、SVD 分解 R/t（4 候选选 1）
#
# 【你需要用到的功能】（都可以从 Day23 / Day25 你自己写过的文件里 import 或复制）
#   - 内参 K
#   - project(X, K, R, t)：3D 点 → 像素
#   - normalize(pixel, K)：像素 → 归一化坐标
#   - solve_E(xh1, xh2)：八点法解本质矩阵 E
#   - decompose_E(E)：SVD 分解出 4 个 (R, t) 候选
#   - triangulate / triangulate_depth：两条反投影光线求 3D 点 / 验证深度
#   注意：复用它还是自己重写，你自己决定。重点是【组装整个流程】。
#
# 【建议步骤】照着这个思路写，但每一步怎么实现由你决定：
#   ① 造 3D 场景：生成 N 个 3D 点，构成一个可辨认的形状（螺旋 / 房子 / 兔子 / 立方体……）
#   ② 定两个相机位姿：相机 1 = 世界（R=I, t=0），相机 2 = 某个 R、t（如绕 y 转 10° + 平移）
#   ③ 投影：用内参 K 把 3D 点投到两个视角，得到两堆像素坐标
#   ④ 归一化：像素坐标 → 归一化坐标
#   ⑤ 八点法：用两堆归一化坐标解 E
#   ⑥ 分解：decompose_E 得到 4 候选，用三角化验证选「深度都 > 0」的那个
#   ⑦ 三角化：用选中的 R、t，把【所有】匹配点三角化回 3D
#   ⑧ 可视化：3D 散点图，把「真实点云」和「重建点云」画在一起对比
#
# 【验收标准】跑通后你应该看到：
#   - 重建点云和真实点云的【形状】一致（可能整体差一个缩放 / 平移 / 旋转）
#   - 两张点云放一起，肉眼能认出是「同一个形状」
#
# 【提示】
#   - 八点法的 t 只有方向、没有尺度，所以重建点云会和真实点云差一个
#     【缩放因子】——这是正常且正确的，不是 bug。你可以顺手观察这个现象，
#     甚至试试把两个点云各自归一化后再比形状。
#   - t 有 ± 符号歧义，选错候选可能重建出「镜像」——用三角化验证来选对。
#   - 3D 点建议 z 在相机前方（比如 2~4），别放到相机后面。
#
# ============================================================
# ↓↓↓ 从这里开始写你自己的代码（下面全是空的）↓↓↓
# ============================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
#内参矩阵
def make_k(fx=300,fy=300,cx=320,cy=240):
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
#做一个三维的球，球心在(2,2,1)，半径是1
def make_ball(N=1000,r=1):
    theta = np.random.uniform(0,2*np.pi,N)
    phi = np.random.uniform(0,np.pi,N)
    x = np.cos(theta)*np.sin(phi)*r+2
    y = np.sin(theta)*np.sin(phi)*r+2
    z = np.cos(phi)*r+1
    points = np.stack([x,y,z],axis=1)
    return points
#给定两个相机的坐标
C1=np.array([0,1,0])
C2=np.array([1,0,0])
#外参数矩阵
def look_at(C, target):
    z = target - C                    # 前方向 = 指向球心
    z = z / np.linalg.norm(z)
    up = np.array([0, 0, 1])          # 世界"上"（或 [0,1,0]，试哪个不歪）
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)   # 三行 = 相机 x/y/z 轴
    return R
#两个相机视角下的点集
def make_Xc1(Xw,R1):
    Xc1=R1@(Xw-C1).T
    return Xc1
def make_Xc2(Xw,R2):
    Xc2=R2@(Xw-C2).T
    return Xc2
#相机拍出来的2D图
def make_dc1(K,Xc1):
    Zc=K@Xc1
    s=Zc[2,:]
    u=Zc[0,:]/s
    v=Zc[1,:]/s
    return np.stack([u,v],axis=0)
def make_dc2(K,Xc2):
    Zc=K@Xc2
    s = Zc[2, :]
    u = Zc[0, :] / s
    v = Zc[1, :] / s
    return np.stack([u, v], axis=0)
#归一化：相机坐标 (X,Y,Z) → 归一化坐标 (X/Z, Y/Z, 1)
def make_xh1(K,Xc1):
    xy = Xc1[:2, :] / Xc1[2, :]            # X/Z, Y/Z
    return np.vstack([xy, np.ones(Xc1.shape[1])])   # 补第三行 1 → (3, N)
def make_xh2(K,Xc2):
    xy = Xc2[:2, :] / Xc2[2, :]
    return np.vstack([xy, np.ones(Xc2.shape[1])])
#回解方程Ae=0
def make_A(x1_norm,x2_norm):
    A=[]
    for (u1, v1, _), (u2, v2, _) in zip(x1_norm, x2_norm):
        row = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1]
        A.append(row)
    return np.array(A)
#得到本质矩阵
def decompose_E(E):
    U,sigma,Vt=np.linalg.svd(E)
    # 修正 U、V 为旋转矩阵（det=+1），否则解出的 R 是反射（det=-1）
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
    R1 = U@W@Vt
    R2 = U@W.T@Vt
    t = U[:,2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]
#回推三维点深度
def triangulate_depth(x1, x2, R, t):
    A = np.stack([R @ x1, -x2], axis=1)   # 3×2
    b = -t
    d = np.linalg.solve(A.T @ A, A.T @ b) # 最小二乘，解出 [d1, d2]
    return d[0], d[1]                      # 返回两个深度
def main():
    K=make_k()
    Ball=make_ball()
    alpha= -np.pi/6
    beta= np.pi/6
    R1 = look_at(C1, np.array([2, 2, 1]))
    R2 = look_at(C2, np.array([2, 2, 1]))
    Xc1=make_Xc1(Ball,R1)
    Xc2=make_Xc2(Ball,R2)
    dc1=make_dc1(K,Xc1)
    dc2=make_dc2(K,Xc2)
    xh1=make_xh1(K,Xc1)
    xh2=make_xh2(K,Xc2)
    A=make_A(xh1.T,xh2.T)
    U,sigma,Vt=np.linalg.svd(A)
    e=Vt[-1]
    E=e.reshape((3,3))     #反解出E
    # 强制秩 2：本质矩阵两个奇异值相等、第三个为 0
    Ue,Se,Vte=np.linalg.svd(E)
    Se[-1]=0
    E=Ue@np.diag(Se)@Vte
    candidates = decompose_E(E)
    best_R,best_t=None,None
    for R,t in candidates:
        ok=True
        for i in range(Ball.shape[0]):
            d1, d2 = triangulate_depth(xh1[:, i], xh2[:, i], R, t)
            if d1 <= 0 or d2 <= 0:
                ok=False
                break
        if ok:
            best_R,best_t=R,t
            break
    pts3d = np.zeros_like(Ball)
    for i in range(Ball.shape[0]):
        d1, _ = triangulate_depth(xh1[:, i], xh2[:, i], best_R, best_t)
        pts3d[i] = d1 * xh1[:, i]  # 深度 × 方向 = 3D 点（在相机1坐标里）

    # ===== 第 8 步：可视化对比 =====
    # 1) 中心化 + 归一化：消除平移和尺度（八点法的 t 尺度不确定）
    def align(pts):
        c = pts - pts.mean(axis=0)                          # 去平移（中心化）
        r = np.sqrt((c ** 2).sum(axis=1)).mean()            # 均方根半径
        return c / r                                        # 去尺度（归一化）

    ball_a = align(Ball)
    pts_a = align(pts3d)

    # 2) Procrustes 旋转对齐：消除重建点云相对真实球的旋转
    Ua, _, Vta = np.linalg.svd(pts_a.T @ ball_a)
    Rot = Ua @ Vta
    pts_a = pts_a @ Rot

    # 3) 量化重建误差（对齐后平均距离，越小越好）
    err = np.sqrt(((ball_a - pts_a) ** 2).sum(axis=1)).mean()
    print("对齐后平均距离（越小越好）:", round(err, 4))

    # 4) 画 3D 叠加对比
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ball_a[:, 0], ball_a[:, 1], ball_a[:, 2], c="royalblue", s=16, label="真实球")
    ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], c="crimson", s=16, label="重建点云", alpha=0.7)
    ax.set_title("两视角重建 3D 点云：真实 vs 重建（已对齐）")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])   # 三轴等比例，球看起来才是圆的
    plt.savefig("可视化图/Day26_两视角重建.png", dpi=150, bbox_inches="tight")
    print("已保存 可视化图/Day26_两视角重建.png")
if __name__=="__main__":
    main()