"""
Configuration file for SVD matrix decomposition homework.
"""
from dataclasses import dataclass


@dataclass
class SVDConfig:
    """Configuration parameters for SVD analysis."""

    # Matrix dimensions
    n_users: int = 100
    n_items: int = 1000

    # Sparsity settings
    density: float = 0.05  # 5% non-zero elements

    # Rating range
    min_rating: int = 1
    max_rating: int = 6  # Exclusive, so range is [1, 5]

    # Dimensionality reduction
    n_components: int = 10  # Number of singular values to keep (r)

    # Random seed for reproducibility
    random_seed: int = 42

    # Display settings
    preview_rows: int = 5
    preview_cols: int = 10
    top_singular_values: int = 20


# Default configuration instance
default_config = SVDConfig()
