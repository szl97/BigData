"""
Similarity analysis utilities for low-dimensional representations.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple, Set

from config import SVDConfig


class SimilarityAnalyzer:
    """Analyzes similarity between rows in low-dimensional space."""

    def __init__(self, config: SVDConfig):
        """
        Initialize the similarity analyzer.

        Args:
            config: Configuration object
        """
        self.config = config

    @staticmethod
    def find_non_intersecting_rows(
        matrix: np.ndarray
    ) -> Tuple[Optional[int], Optional[int], Optional[Set], Optional[Set]]:
        """
        Find two rows with no overlapping non-zero elements.

        Args:
            matrix: Input matrix to analyze

        Returns:
            Tuple of (row_i, row_j, nonzero_i, nonzero_j) or (None, None, None, None)
        """
        n_rows = matrix.shape[0]

        for i in range(n_rows):
            nonzero_i = set(np.where(matrix[i, :] != 0)[0])
            if len(nonzero_i) == 0:
                continue

            for j in range(i + 1, n_rows):
                nonzero_j = set(np.where(matrix[j, :] != 0)[0])
                if len(nonzero_j) == 0:
                    continue

                # Check if intersection is empty
                if len(nonzero_i & nonzero_j) == 0:
                    return i, j, nonzero_i, nonzero_j

        return None, None, None, None

    @staticmethod
    def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity value
        """
        return cosine_similarity([vec1], [vec2])[0, 0]

    @staticmethod
    def compute_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Euclidean distance value
        """
        return np.linalg.norm(vec1 - vec2)

    def analyze_row_similarity(
        self,
        original_matrix: np.ndarray,
        lowdim_matrix: np.ndarray,
        row_i: int,
        row_j: int,
        nonzero_i: Set[int],
        nonzero_j: Set[int]
    ) -> dict:
        """
        Analyze similarity between two rows in both original and low-dimensional space.

        Args:
            original_matrix: Original high-dimensional matrix
            lowdim_matrix: Low-dimensional representation matrix (e.g., U_r)
            row_i: First row index
            row_j: Second row index
            nonzero_i: Set of non-zero indices in row i
            nonzero_j: Set of non-zero indices in row j

        Returns:
            Dictionary with similarity metrics
        """
        # Low-dimensional representations
        vec_i = lowdim_matrix[row_i, :]
        vec_j = lowdim_matrix[row_j, :]

        # Compute similarities
        cosine_sim = self.compute_cosine_similarity(vec_i, vec_j)
        euclidean_dist = self.compute_euclidean_distance(vec_i, vec_j)
        dot_product = np.dot(vec_i, vec_j)

        return {
            'row_i': row_i,
            'row_j': row_j,
            'nonzero_i_count': len(nonzero_i),
            'nonzero_j_count': len(nonzero_j),
            'intersection_size': len(nonzero_i & nonzero_j),
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'dot_product': dot_product,
            'vec_i': vec_i,
            'vec_j': vec_j
        }

    def print_similarity_analysis(
        self,
        original_matrix: np.ndarray,
        analysis_result: dict
    ) -> None:
        """
        Print formatted similarity analysis results.

        Args:
            original_matrix: Original matrix for displaying samples
            analysis_result: Result dictionary from analyze_row_similarity
        """
        i = analysis_result['row_i']
        j = analysis_result['row_j']

        print(f"找到的两行: 行{i} 和 行{j}")
        print(f"行{i}的非零元素位置数量: {analysis_result['nonzero_i_count']}")
        print(f"行{j}的非零元素位置数量: {analysis_result['nonzero_j_count']}")
        print(f"交集大小: {analysis_result['intersection_size']} (无交集)")

        print(f"\n原始空间中:")
        print(f"行{i}的前20个元素: {original_matrix[i, :20]}")
        print(f"行{j}的前20个元素: {original_matrix[j, :20]}")

        print(f"\n低维空间表示 (r={self.config.n_components}):")
        print(f"行{i}的低维向量: {analysis_result['vec_i']}")
        print(f"行{j}的低维向量: {analysis_result['vec_j']}")

        print(f"\n相似度计算结果:")
        print(f"余弦相似度: {analysis_result['cosine_similarity']:.6f}")
        print(f"欧氏距离: {analysis_result['euclidean_distance']:.6f}")
        print(f"点积: {analysis_result['dot_product']:.6f}")
