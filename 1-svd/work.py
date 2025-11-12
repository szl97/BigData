"""
SVD矩阵分解与降维作业
实现基于SVD的稀疏矩阵降维，模拟推荐系统中的用户-商家评分分析
"""

import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Set
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SVDAnalyzer:
    """SVD分解与降维分析器"""

    def __init__(self, n_users: int = 100, n_items: int = 1000,
                 density: float = 0.05, rating_range: Tuple[int, int] = (1, 6),
                 random_state: int = 42):
        """
        初始化SVD分析器

        参数:
            n_users: 用户数量
            n_items: 商家数量
            density: 矩阵密度（非零元素比例）
            rating_range: 评分范围 (min, max)
            random_state: 随机种子
        """
        self.n_users = n_users
        self.n_items = n_items
        self.density = density
        self.rating_range = rating_range
        self.random_state = random_state

        # 初始化矩阵和分解结果
        self.A = None
        self.U = None
        self.sigma = None
        self.Vt = None
        self.U_r = None
        self.sigma_r = None
        self.Vt_r = None
        self.A_reduced = None

    def generate_sparse_matrix(self) -> np.ndarray:
        """
        生成稀疏评分矩阵

        返回:
            稀疏矩阵 (n_users × n_items)
        """
        print(f"\n{'='*60}")
        print("【步骤1】构造稀疏矩阵")
        print(f"{'='*60}")

        # 生成稀疏矩阵
        sparse_matrix = sparse_random(
            self.n_users,
            self.n_items,
            density=self.density,
            format='csr',
            random_state=self.random_state,
            data_rvs=lambda s: np.random.randint(*self.rating_range, size=s)
        )
        self.A = sparse_matrix.toarray()

        # 输出矩阵信息
        n_nonzero = np.count_nonzero(self.A)
        sparsity = 1 - n_nonzero / self.A.size

        print(f"矩阵形状: {self.A.shape}")
        print(f"非零元素数量: {n_nonzero:,}")
        print(f"稀疏度: {sparsity:.2%}")
        print(f"密度: {self.density:.2%}")
        print(f"评分范围: [{self.rating_range[0]}, {self.rating_range[1]-1}]")
        print(f"\n矩阵A的前5×10示例:")
        print(self.A[:5, :10])

        return self.A

    def perform_svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行SVD分解

        返回:
            U, sigma, Vt 三个矩阵
        """
        print(f"\n{'='*60}")
        print("【步骤2】SVD分解")
        print(f"{'='*60}")

        if self.A is None:
            raise ValueError("请先生成稀疏矩阵")

        # 执行SVD分解: A = U @ Σ @ V^T
        self.U, self.sigma, self.Vt = svd(self.A, full_matrices=False)

        print(f"U矩阵形状: {self.U.shape}")
        print(f"奇异值向量形状: {self.sigma.shape}")
        print(f"V^T矩阵形状: {self.Vt.shape}")
        print(f"\n前20个奇异值:")
        print(np.array2string(self.sigma[:20], precision=4, suppress_small=True))

        return self.U, self.sigma, self.Vt

    def reduce_dimension(self, r: int = 10) -> np.ndarray:
        """
        使用前r个奇异值进行降维

        参数:
            r: 保留的奇异值数量

        返回:
            降维后的矩阵 A_reduced
        """
        print(f"\n{'='*60}")
        print(f"【步骤3】降维 (r={r})")
        print(f"{'='*60}")

        if self.U is None or self.sigma is None or self.Vt is None:
            raise ValueError("请先执行SVD分解")

        # 取前r个分量
        self.U_r = self.U[:, :r]
        self.sigma_r = self.sigma[:r]
        self.Vt_r = self.Vt[:r, :]

        # 重构矩阵
        Sigma_r = np.diag(self.sigma_r)
        self.A_reduced = self.U_r @ Sigma_r @ self.Vt_r

        print(f"降维后的矩阵维度:")
        print(f"  U_r: {self.U_r.shape}")
        print(f"  Sigma_r: {Sigma_r.shape}")
        print(f"  Vt_r: {self.Vt_r.shape}")
        print(f"  A_reduced: {self.A_reduced.shape}")

        # 计算重构误差
        reconstruction_error = np.linalg.norm(self.A - self.A_reduced, 'fro')
        relative_error = reconstruction_error / np.linalg.norm(self.A, 'fro')

        print(f"\n重构误差分析:")
        print(f"  Frobenius范数: {reconstruction_error:.4f}")
        print(f"  相对误差: {relative_error:.4%}")

        # 能量保留率
        energy_retained = np.sum(self.sigma_r**2) / np.sum(self.sigma**2)
        print(f"  前{r}个奇异值保留的能量比例: {energy_retained:.2%}")

        return self.A_reduced

    def find_non_intersecting_rows(self) -> Tuple[Optional[int], Optional[int],
                                                    Optional[Set], Optional[Set]]:
        """
        高效查找两行，使得它们的非零元素位置没有交集

        返回:
            (行索引i, 行索引j, 行i的非零位置集合, 行j的非零位置集合)
            如果未找到，返回 (None, None, None, None)
        """
        print(f"\n{'='*60}")
        print("【步骤4】查找无交集的行")
        print(f"{'='*60}")

        if self.A is None:
            raise ValueError("请先生成稀疏矩阵")

        # 预计算所有行的非零位置
        nonzero_positions = []
        for i in range(self.n_users):
            nonzero_i = set(np.where(self.A[i, :] != 0)[0])
            if len(nonzero_i) > 0:
                nonzero_positions.append((i, nonzero_i))

        print(f"有效行数（含非零元素）: {len(nonzero_positions)}")

        # 查找无交集的两行
        for idx1 in range(len(nonzero_positions)):
            i, nonzero_i = nonzero_positions[idx1]

            for idx2 in range(idx1 + 1, len(nonzero_positions)):
                j, nonzero_j = nonzero_positions[idx2]

                # 检查交集是否为空
                if len(nonzero_i & nonzero_j) == 0:
                    print(f"✓ 找到符合条件的两行: 行{i} 和 行{j}")
                    print(f"  行{i}的非零元素位置数量: {len(nonzero_i)}")
                    print(f"  行{j}的非零元素位置数量: {len(nonzero_j)}")
                    print(f"  交集大小: 0 (完全无交集)")
                    return i, j, nonzero_i, nonzero_j

        print("✗ 未找到满足条件的两行")
        print("  提示: 可以尝试增加密度或调整随机种子")
        return None, None, None, None

    def compute_similarity(self, row_i: int, row_j: int) -> dict:
        """
        计算两行在低维空间的相似度

        参数:
            row_i: 第一行的索引
            row_j: 第二行的索引

        返回:
            包含各种相似度度量的字典
        """
        print(f"\n{'='*60}")
        print(f"【步骤5】计算低维空间相似度")
        print(f"{'='*60}")

        if self.U_r is None:
            raise ValueError("请先执行降维操作")

        # 原始空间中的向量
        print(f"\n原始空间 (维度={self.n_items}):")
        print(f"  行{row_i}的前20个元素: {self.A[row_i, :20]}")
        print(f"  行{row_j}的前20个元素: {self.A[row_j, :20]}")

        # 低维空间中的表示
        user_i_lowdim = self.U_r[row_i, :]
        user_j_lowdim = self.U_r[row_j, :]

        r = self.U_r.shape[1]
        print(f"\n低维空间 (维度={r}):")
        print(f"  行{row_i}的低维向量: {user_i_lowdim}")
        print(f"  行{row_j}的低维向量: {user_j_lowdim}")

        # 计算多种相似度度量
        cosine_sim = cosine_similarity([user_i_lowdim], [user_j_lowdim])[0, 0]
        euclidean_dist = np.linalg.norm(user_i_lowdim - user_j_lowdim)
        dot_product = np.dot(user_i_lowdim, user_j_lowdim)

        # L1距离（曼哈顿距离）
        manhattan_dist = np.sum(np.abs(user_i_lowdim - user_j_lowdim))

        print(f"\n相似度度量结果:")
        print(f"  余弦相似度: {cosine_sim:.6f}")
        print(f"  欧氏距离: {euclidean_dist:.6f}")
        print(f"  曼哈顿距离: {manhattan_dist:.6f}")
        print(f"  点积: {dot_product:.6f}")

        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'manhattan_distance': manhattan_dist,
            'dot_product': dot_product
        }

    def visualize_results(self, r: int = 10, save_dir: str = './'):
        """
        可视化SVD分解和降维结果

        参数:
            r: 降维维度
            save_dir: 图片保存目录
        """
        if self.sigma is None:
            raise ValueError("请先执行SVD分解")

        print(f"\n{'='*60}")
        print("【步骤6】生成可视化图表")
        print(f"{'='*60}")

        # 创建大画布
        fig = plt.figure(figsize=(16, 12))

        # 1. 奇异值分布图
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(range(1, len(self.sigma) + 1), self.sigma, 'b-', linewidth=2)
        plt.axvline(x=r, color='r', linestyle='--', label=f'r={r}')
        plt.xlabel('Singular Value Index', fontsize=12)
        plt.ylabel('Singular Value', fontsize=12)
        plt.title('Singular Value Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 2. 奇异值累积能量图
        ax2 = plt.subplot(2, 3, 2)
        cumulative_energy = np.cumsum(self.sigma**2) / np.sum(self.sigma**2)
        plt.plot(range(1, len(self.sigma) + 1), cumulative_energy * 100, 'g-', linewidth=2)
        plt.axvline(x=r, color='r', linestyle='--', label=f'r={r}')
        plt.axhline(y=cumulative_energy[r-1] * 100, color='r', linestyle=':', alpha=0.5)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Energy (%)', fontsize=12)
        plt.title('Cumulative Energy Retention', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 3. 重构误差随r变化
        ax3 = plt.subplot(2, 3, 3)
        r_values = range(1, min(51, len(self.sigma) + 1))
        reconstruction_errors = []

        for r_val in r_values:
            U_temp = self.U[:, :r_val]
            sigma_temp = self.sigma[:r_val]
            Vt_temp = self.Vt[:r_val, :]
            A_temp = U_temp @ np.diag(sigma_temp) @ Vt_temp
            error = np.linalg.norm(self.A - A_temp, 'fro') / np.linalg.norm(self.A, 'fro')
            reconstruction_errors.append(error * 100)

        plt.plot(r_values, reconstruction_errors, 'purple', linewidth=2)
        plt.axvline(x=r, color='r', linestyle='--', label=f'r={r}')
        plt.xlabel('Number of Components (r)', fontsize=12)
        plt.ylabel('Relative Error (%)', fontsize=12)
        plt.title('Reconstruction Error vs r', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 4. 原始矩阵热力图（采样）
        ax4 = plt.subplot(2, 3, 4)
        sample_users = min(50, self.n_users)
        sample_items = min(100, self.n_items)
        sns.heatmap(self.A[:sample_users, :sample_items],
                   cmap='YlOrRd', cbar_kws={'label': 'Rating'},
                   xticklabels=False, yticklabels=False)
        plt.xlabel(f'Items (showing first {sample_items})', fontsize=12)
        plt.ylabel(f'Users (showing first {sample_users})', fontsize=12)
        plt.title('Original Sparse Matrix', fontsize=14, fontweight='bold')

        # 5. 降维重构矩阵热力图
        ax5 = plt.subplot(2, 3, 5)
        if self.A_reduced is not None:
            sns.heatmap(self.A_reduced[:sample_users, :sample_items],
                       cmap='YlOrRd', cbar_kws={'label': 'Rating'},
                       xticklabels=False, yticklabels=False)
            plt.xlabel(f'Items (showing first {sample_items})', fontsize=12)
            plt.ylabel(f'Users (showing first {sample_users})', fontsize=12)
            plt.title(f'Reconstructed Matrix (r={r})', fontsize=14, fontweight='bold')

        # 6. 统计信息面板
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # 准备统计信息文本
        n_nonzero = np.count_nonzero(self.A)
        sparsity = 1 - n_nonzero / self.A.size
        energy_retained = np.sum(self.sigma_r**2) / np.sum(self.sigma**2)
        reconstruction_error = np.linalg.norm(self.A - self.A_reduced, 'fro')
        relative_error = reconstruction_error / np.linalg.norm(self.A, 'fro')

        stats_text = f"""
        SVD Analysis Summary
        {'='*40}

        Matrix Properties:
          • Shape: {self.A.shape[0]} × {self.A.shape[1]}
          • Non-zero elements: {n_nonzero:,}
          • Sparsity: {sparsity:.2%}
          • Density: {self.density:.2%}

        SVD Decomposition:
          • Total singular values: {len(self.sigma)}
          • Components retained: {r}
          • Energy retained: {energy_retained:.2%}

        Reconstruction Quality:
          • Frobenius error: {reconstruction_error:.4f}
          • Relative error: {relative_error:.4%}

        Dimension Reduction:
          • Original dim: {self.n_items}
          • Reduced dim: {r}
          • Compression ratio: {self.n_items/r:.1f}x
        """

        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图片
        output_path = f'{save_dir}/svd_analysis_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图表已保存: {output_path}")

        plt.show()

        return output_path


def main():
    """主函数：执行完整的SVD分析流程"""

    print("=" * 60)
    print("作业1：SVD矩阵分解与降维")
    print("=" * 60)

    # 初始化分析器
    analyzer = SVDAnalyzer(
        n_users=100,
        n_items=1000,
        density=0.05,
        rating_range=(1, 6),
        random_state=42
    )

    # 1. 生成稀疏矩阵
    A = analyzer.generate_sparse_matrix()

    # 2. 执行SVD分解
    U, sigma, Vt = analyzer.perform_svd()

    # 3. 降维（r=10）
    r = 10
    A_reduced = analyzer.reduce_dimension(r=r)

    # 4. 查找无交集的两行
    i, j, nonzero_i, nonzero_j = analyzer.find_non_intersecting_rows()

    # 5. 计算低维空间相似度
    if i is not None and j is not None:
        similarity_metrics = analyzer.compute_similarity(i, j)
    else:
        print("\n⚠ 由于未找到无交集的行，跳过相似度计算")

    # 6. 生成可视化
    try:
        analyzer.visualize_results(r=r, save_dir='./')
    except Exception as e:
        print(f"\n⚠ 可视化生成失败: {e}")
        print("  提示: 请确保已安装 matplotlib 和 seaborn")

    print("\n" + "=" * 60)
    print("作业完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
