import numpy as np
from scipy.sparse import random
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
import time

# 设置随机种子，保证结果可复现
np.random.seed(42)


def create_sparse_matrix(n_users, n_items, density, random_state=42):
    """
    创建稀疏用户-商家评分矩阵

    参数:
        n_users: 用户数量
        n_items: 商家数量
        density: 矩阵密度 (非零元素比例)
        random_state: 随机种子

    返回:
        密集矩阵形式的评分矩阵
    """
    A_sparse = random(
        n_users, n_items,
        density=density,
        format='csr',
        random_state=random_state,
        data_rvs=lambda s: np.random.randint(1, 6, size=s)
    )
    return A_sparse.toarray()


def perform_svd_decomposition(A, r):
    """
    对矩阵进行SVD分解并降维

    参数:
        A: 输入矩阵
        r: 保留的奇异值数量

    返回:
        U_r, sigma_r, Vt_r: 降维后的矩阵
        sigma: 所有奇异值
        A_reduced: 重构的降维矩阵
    """
    # 执行SVD分解: A = U @ Σ @ V^T
    U, sigma, Vt = svd(A, full_matrices=False)

    # 降维: 取前r个分量
    U_r = U[:, :r]
    sigma_r = sigma[:r]
    Vt_r = Vt[:r, :]

    # 重构降维后的矩阵
    Sigma_r = np.diag(sigma_r)
    A_reduced = U_r @ Sigma_r @ Vt_r

    return U_r, sigma_r, Vt_r, sigma, A_reduced


def find_non_intersecting_rows_optimized(matrix):
    """
    高效查找两行，使得它们的非零元素位置没有交集

    优化策略:
    1. 使用向量化操作获取所有非零位置
    2. 预先计算每行的非零列索引
    3. 使用NumPy的快速集合操作

    参数:
        matrix: 输入矩阵

    返回:
        i, j: 满足条件的两行索引
        nonzero_i, nonzero_j: 两行的非零元素位置集合
    """
    n_rows = matrix.shape[0]

    # 预计算所有行的非零列索引 (向量化操作)
    nonzero_cols = []
    for i in range(n_rows):
        cols = np.where(matrix[i, :] != 0)[0]
        if len(cols) > 0:
            nonzero_cols.append((i, set(cols)))

    # 快速查找无交集的行对
    for idx1 in range(len(nonzero_cols)):
        i, cols_i = nonzero_cols[idx1]

        for idx2 in range(idx1 + 1, len(nonzero_cols)):
            j, cols_j = nonzero_cols[idx2]

            # 使用集合的isdisjoint方法,比检查交集长度更快
            if cols_i.isdisjoint(cols_j):
                return i, j, cols_i, cols_j

    return None, None, None, None


def calculate_similarity_metrics(vec1, vec2):
    """
    计算两个向量的多种相似度指标

    参数:
        vec1, vec2: 两个向量

    返回:
        包含余弦相似度、欧氏距离和点积的字典
    """
    # 余弦相似度
    cos_sim = cosine_similarity([vec1], [vec2])[0, 0]

    # 欧氏距离
    euclidean_dist = np.linalg.norm(vec1 - vec2)

    # 点积
    dot_prod = np.dot(vec1, vec2)

    return {
        'cosine_similarity': cos_sim,
        'euclidean_distance': euclidean_dist,
        'dot_product': dot_prod
    }


def print_section(title, width=60):
    """打印格式化的章节标题"""
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}")


def print_subsection(title, width=60):
    """打印格式化的子章节标题"""
    print(f"\n{title}")
    print(f"{'-' * width}")


def main():
    """主函数：执行SVD矩阵分解与降维作业"""

    print_section("作业：SVD矩阵分解与降维")

    # ===== 步骤1: 构造稀疏矩阵 =====
    print_subsection("【步骤1】构造稀疏矩阵")

    # 参数设置
    n_users = 100
    n_items = 1000
    density = 0.05
    r = 10  # 降维维度

    start_time = time.time()
    A = create_sparse_matrix(n_users, n_items, density)
    matrix_time = time.time() - start_time

    print(f"矩阵形状: {A.shape}")
    print(f"非零元素数量: {np.count_nonzero(A)}")
    print(f"稀疏度: {1 - np.count_nonzero(A) / A.size:.2%}")
    print(f"密度: {np.count_nonzero(A) / A.size:.2%}")
    print(f"生成时间: {matrix_time:.4f}秒")
    print(f"\n矩阵A的前5×10示例:")
    print(A[:5, :10])

    # ===== 步骤2(a): SVD分解与降维 =====
    print_subsection("【步骤2(a)】SVD分解与降维")

    start_time = time.time()
    U_r, sigma_r, Vt_r, sigma, A_reduced = perform_svd_decomposition(A, r)
    svd_time = time.time() - start_time

    print(f"U矩阵形状: {(A.shape[0], A.shape[0])}")
    print(f"奇异值向量形状: {(min(A.shape),)}")
    print(f"V^T矩阵形状: {(min(A.shape), A.shape[1])}")
    print(f"SVD分解时间: {svd_time:.4f}秒")
    print(f"\n前20个奇异值:")
    print(sigma[:20])

    # 降维结果
    print(f"\n降维后的矩阵维度 (r={r}):")
    print(f"U_r: {U_r.shape}")
    print(f"Sigma_r: {(r, r)}")
    print(f"Vt_r: {Vt_r.shape}")
    print(f"A_reduced: {A_reduced.shape}")

    # 误差分析
    reconstruction_error = np.linalg.norm(A - A_reduced, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(A, 'fro')
    energy_retained = np.sum(sigma_r**2) / np.sum(sigma**2)

    print(f"\n重构误差 (Frobenius范数): {reconstruction_error:.4f}")
    print(f"相对误差: {relative_error:.4%}")
    print(f"前{r}个奇异值保留的能量比例: {energy_retained:.2%}")

    # ===== 步骤2(b): 找到两行无交集的行并计算相似度 =====
    print_subsection("【步骤2(b)】选择两行并计算低维相似度")

    start_time = time.time()
    i, j, nonzero_i, nonzero_j = find_non_intersecting_rows_optimized(A)
    search_time = time.time() - start_time

    if i is not None and j is not None:
        print(f"找到的两行: 行{i} 和 行{j}")
        print(f"行{i}的非零元素位置数量: {len(nonzero_i)}")
        print(f"行{j}的非零元素位置数量: {len(nonzero_j)}")
        print(f"交集大小: 0 (无交集)")
        print(f"查找时间: {search_time:.4f}秒")

        # 原始空间中的向量
        print(f"\n原始空间中:")
        print(f"行{i}的前20个元素: {A[i, :20]}")
        print(f"行{j}的前20个元素: {A[j, :20]}")

        # 低维空间中的表示
        user_i_lowdim = U_r[i, :]
        user_j_lowdim = U_r[j, :]

        print(f"\n低维空间表示 (r={r}):")
        print(f"行{i}的低维向量: {user_i_lowdim}")
        print(f"行{j}的低维向量: {user_j_lowdim}")

        # 计算相似度指标
        metrics = calculate_similarity_metrics(user_i_lowdim, user_j_lowdim)

        print(f"\n相似度计算结果:")
        print(f"余弦相似度: {metrics['cosine_similarity']:.6f}")
        print(f"欧氏距离: {metrics['euclidean_distance']:.6f}")
        print(f"点积: {metrics['dot_product']:.6f}")

        # 解释相似度的意义
        print(f"\n相似度分析:")
        if abs(metrics['cosine_similarity']) > 0.5:
            print("→ 余弦相似度较高，说明两个用户在低维空间中有相似的偏好方向")
        elif abs(metrics['cosine_similarity']) < 0.1:
            print("→ 余弦相似度较低，说明两个用户的偏好方向差异较大")
        else:
            print("→ 余弦相似度中等，说明两个用户有一定程度的相似性")

    else:
        print("未找到满足条件的两行（非零元素位置完全不相交）")
        print("这在稀疏矩阵中可能发生，可以尝试增加密度或调整随机种子")
        print(f"查找时间: {search_time:.4f}秒")

    # ===== 性能总结 =====
    print_subsection("【性能总结】")
    total_time = matrix_time + svd_time + search_time
    print(f"矩阵生成时间: {matrix_time:.4f}秒")
    print(f"SVD分解时间: {svd_time:.4f}秒")
    print(f"行查找时间: {search_time:.4f}秒")
    print(f"总运行时间: {total_time:.4f}秒")

    print_section("作业完成！")

    return {
        'matrix': A,
        'U_r': U_r,
        'sigma_r': sigma_r,
        'Vt_r': Vt_r,
        'reconstruction_error': reconstruction_error,
        'energy_retained': energy_retained
    }


if __name__ == "__main__":
    results = main()
