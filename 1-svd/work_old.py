import numpy as np
from scipy.sparse import random
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity

# 设置随机种子，保证结果可复现
np.random.seed(42)

print("=" * 60)
print("作业：SVD矩阵分解与降维")
print("=" * 60)

# 1. 构造100×1000的稀疏矩阵
print("\n【步骤1】构造稀疏矩阵")
print("-" * 60)

# 稀疏性设置：只有5%的元素非零（密度=0.05）
density = 0.05
n_users = 100  # 用户数
n_items = 1000  # 商家数

# 使用scipy.sparse生成稀疏矩阵，然后转为密集矩阵
# 评分范围设为1-5分
A = random(n_users, n_items, density=density, format='csr',
           random_state=42, data_rvs=lambda s: np.random.randint(1, 6, size=s))
A = A.toarray()

print(f"矩阵形状: {A.shape}")
print(f"非零元素数量: {np.count_nonzero(A)}")
print(f"稀疏度: {1 - np.count_nonzero(A) / A.size:.2%}")
print(f"密度: {np.count_nonzero(A) / A.size:.2%}")
print(f"\n矩阵A的前5×10示例:")
print(A[:5, :10])

# (a) SVD分解
print("\n【步骤2(a)】SVD分解与降维")
print("-" * 60)

# 执行SVD分解: A = U @ Σ @ V^T
U, sigma, Vt = svd(A, full_matrices=False)

print(f"U矩阵形状: {U.shape}")
print(f"奇异值向量形状: {sigma.shape}")
print(f"V^T矩阵形状: {Vt.shape}")
print(f"\n前20个奇异值:")
print(sigma[:20])

# r=10降维
r = 10
U_r = U[:, :r]  # 取前r列
sigma_r = sigma[:r]  # 取前r个奇异值
Vt_r = Vt[:r, :]  # 取前r行

# 构造降维后的矩阵
Sigma_r = np.diag(sigma_r)
A_reduced = U_r @ Sigma_r @ Vt_r

print(f"\n降维后的矩阵维度:")
print(f"U_r: {U_r.shape}")
print(f"Sigma_r: {Sigma_r.shape}")
print(f"Vt_r: {Vt_r.shape}")
print(f"A_reduced: {A_reduced.shape}")

# 计算重构误差
reconstruction_error = np.linalg.norm(A - A_reduced, 'fro')
relative_error = reconstruction_error / np.linalg.norm(A, 'fro')
print(f"\n重构误差 (Frobenius范数): {reconstruction_error:.4f}")
print(f"相对误差: {relative_error:.4%}")

# 能量保留率
energy_retained = np.sum(sigma_r**2) / np.sum(sigma**2)
print(f"前{r}个奇异值保留的能量比例: {energy_retained:.2%}")

# (b) 找到两行无交集的行，计算低维相似度
print("\n【步骤2(b)】选择两行并计算低维相似度")
print("-" * 60)

# 寻找两行，使得非零元素位置没有交集
def find_non_intersecting_rows(matrix):
    """找到两行，使得它们的非零元素位置没有交集"""
    n_rows = matrix.shape[0]

    for i in range(n_rows):
        nonzero_i = set(np.where(matrix[i, :] != 0)[0])
        if len(nonzero_i) == 0:
            continue

        for j in range(i + 1, n_rows):
            nonzero_j = set(np.where(matrix[j, :] != 0)[0])
            if len(nonzero_j) == 0:
                continue

            # 检查交集是否为空
            if len(nonzero_i & nonzero_j) == 0:
                return i, j, nonzero_i, nonzero_j

    return None, None, None, None

i, j, nonzero_i, nonzero_j = find_non_intersecting_rows(A)

if i is not None and j is not None:
    print(f"找到的两行: 行{i} 和 行{j}")
    print(f"行{i}的非零元素位置数量: {len(nonzero_i)}")
    print(f"行{j}的非零元素位置数量: {len(nonzero_j)}")
    print(f"交集大小: 0 (无交集)")

    # 在原始空间中的向量
    print(f"\n原始空间中:")
    print(f"行{i}的前20个元素: {A[i, :20]}")
    print(f"行{j}的前20个元素: {A[j, :20]}")

    # 在低维空间中的表示 (U_r的对应行)
    user_i_lowdim = U_r[i, :]
    user_j_lowdim = U_r[j, :]

    print(f"\n低维空间表示 (r={r}):")
    print(f"行{i}的低维向量: {user_i_lowdim}")
    print(f"行{j}的低维向量: {user_j_lowdim}")

    # 计算余弦相似度
    similarity = cosine_similarity([user_i_lowdim], [user_j_lowdim])[0, 0]

    # 计算欧氏距离
    euclidean_dist = np.linalg.norm(user_i_lowdim - user_j_lowdim)

    print(f"\n相似度计算结果:")
    print(f"余弦相似度: {similarity:.6f}")
    print(f"欧氏距离: {euclidean_dist:.6f}")

    # 额外信息：点积
    dot_product = np.dot(user_i_lowdim, user_j_lowdim)
    print(f"点积: {dot_product:.6f}")

else:
    print("未找到满足条件的两行（非零元素位置完全不相交）")
    print("这在稀疏矩阵中可能发生，可以尝试增加密度或调整随机种子")

print("\n" + "=" * 60)
print("作业完成！")
print("=" * 60)