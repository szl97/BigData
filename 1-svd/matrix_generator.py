"""
Matrix generation utilities for sparse user-item rating matrices.
"""
import numpy as np
from scipy.sparse import random
from typing import Tuple

from config import SVDConfig


class SparseMatrixGenerator:
    """Generates sparse rating matrices for recommendation system simulation."""

    def __init__(self, config: SVDConfig):
        """
        Initialize the matrix generator.

        Args:
            config: Configuration object with matrix parameters
        """
        self.config = config
        np.random.seed(config.random_seed)

    def generate_matrix(self) -> np.ndarray:
        """
        Generate a sparse user-item rating matrix.

        Returns:
            Dense numpy array of shape (n_users, n_items) with ratings
        """
        # Generate sparse matrix with scipy
        sparse_matrix = random(
            self.config.n_users,
            self.config.n_items,
            density=self.config.density,
            format='csr',
            random_state=self.config.random_seed,
            data_rvs=lambda s: np.random.randint(
                self.config.min_rating,
                self.config.max_rating,
                size=s
            )
        )

        # Convert to dense array
        return sparse_matrix.toarray()

    @staticmethod
    def get_matrix_info(matrix: np.ndarray) -> dict:
        """
        Get statistical information about the matrix.

        Args:
            matrix: Input matrix

        Returns:
            Dictionary with matrix statistics
        """
        n_nonzero = np.count_nonzero(matrix)
        total_elements = matrix.size

        return {
            'shape': matrix.shape,
            'n_nonzero': n_nonzero,
            'total_elements': total_elements,
            'density': n_nonzero / total_elements,
            'sparsity': 1 - (n_nonzero / total_elements)
        }

    def print_matrix_info(self, matrix: np.ndarray) -> None:
        """
        Print formatted information about the matrix.

        Args:
            matrix: Input matrix to analyze
        """
        info = self.get_matrix_info(matrix)

        print(f"矩阵形状: {info['shape']}")
        print(f"非零元素数量: {info['n_nonzero']}")
        print(f"稀疏度: {info['sparsity']:.2%}")
        print(f"密度: {info['density']:.2%}")
        print(f"\n矩阵A的前{self.config.preview_rows}×{self.config.preview_cols}示例:")
        print(matrix[:self.config.preview_rows, :self.config.preview_cols])
