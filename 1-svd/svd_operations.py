"""
SVD decomposition and dimensionality reduction operations.
"""
import numpy as np
from scipy.linalg import svd
from typing import Tuple, NamedTuple

from config import SVDConfig


class SVDResult(NamedTuple):
    """Container for SVD decomposition results."""
    U: np.ndarray
    sigma: np.ndarray
    Vt: np.ndarray
    U_r: np.ndarray
    sigma_r: np.ndarray
    Vt_r: np.ndarray
    A_reduced: np.ndarray


class SVDProcessor:
    """Performs SVD decomposition and dimensionality reduction."""

    def __init__(self, config: SVDConfig):
        """
        Initialize the SVD processor.

        Args:
            config: Configuration object with SVD parameters
        """
        self.config = config

    def decompose(self, matrix: np.ndarray) -> SVDResult:
        """
        Perform SVD decomposition and dimensionality reduction.

        Args:
            matrix: Input matrix to decompose

        Returns:
            SVDResult containing all decomposition components
        """
        # Full SVD decomposition
        U, sigma, Vt = svd(matrix, full_matrices=False)

        # Dimensionality reduction to r components
        r = self.config.n_components
        U_r = U[:, :r]
        sigma_r = sigma[:r]
        Vt_r = Vt[:r, :]

        # Reconstruct reduced matrix
        Sigma_r = np.diag(sigma_r)
        A_reduced = U_r @ Sigma_r @ Vt_r

        return SVDResult(
            U=U,
            sigma=sigma,
            Vt=Vt,
            U_r=U_r,
            sigma_r=sigma_r,
            Vt_r=Vt_r,
            A_reduced=A_reduced
        )

    def print_svd_info(self, result: SVDResult) -> None:
        """
        Print formatted SVD decomposition information.

        Args:
            result: SVD decomposition result
        """
        print(f"U矩阵形状: {result.U.shape}")
        print(f"奇异值向量形状: {result.sigma.shape}")
        print(f"V^T矩阵形状: {result.Vt.shape}")
        print(f"\n前{self.config.top_singular_values}个奇异值:")
        print(result.sigma[:self.config.top_singular_values])

        print(f"\n降维后的矩阵维度:")
        print(f"U_r: {result.U_r.shape}")
        print(f"Sigma_r: {np.diag(result.sigma_r).shape}")
        print(f"Vt_r: {result.Vt_r.shape}")
        print(f"A_reduced: {result.A_reduced.shape}")

    def compute_reconstruction_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute reconstruction error metrics.

        Args:
            original: Original matrix
            reconstructed: Reconstructed matrix after dimensionality reduction

        Returns:
            Tuple of (absolute_error, relative_error)
        """
        absolute_error = np.linalg.norm(original - reconstructed, 'fro')
        relative_error = absolute_error / np.linalg.norm(original, 'fro')

        return absolute_error, relative_error

    def compute_energy_retention(
        self,
        sigma: np.ndarray,
        sigma_r: np.ndarray
    ) -> float:
        """
        Compute the proportion of energy retained after reduction.

        Args:
            sigma: Full singular values
            sigma_r: Reduced singular values

        Returns:
            Energy retention ratio
        """
        return np.sum(sigma_r**2) / np.sum(sigma**2)

    def print_reconstruction_metrics(
        self,
        original: np.ndarray,
        result: SVDResult
    ) -> None:
        """
        Print reconstruction quality metrics.

        Args:
            original: Original matrix
            result: SVD decomposition result
        """
        abs_error, rel_error = self.compute_reconstruction_error(
            original,
            result.A_reduced
        )
        energy = self.compute_energy_retention(result.sigma, result.sigma_r)

        print(f"\n重构误差 (Frobenius范数): {abs_error:.4f}")
        print(f"相对误差: {rel_error:.4%}")
        print(f"前{self.config.n_components}个奇异值保留的能量比例: {energy:.2%}")
