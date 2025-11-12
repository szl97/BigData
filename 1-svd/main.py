"""
Main script for SVD Matrix Decomposition and Dimensionality Reduction homework.

This script demonstrates:
1. Sparse matrix construction
2. SVD decomposition
3. Dimensionality reduction
4. Similarity analysis in low-dimensional space
"""
from config import default_config
from matrix_generator import SparseMatrixGenerator
from svd_operations import SVDProcessor
from similarity_analyzer import SimilarityAnalyzer


def print_section_header(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def main():
    """Main execution function."""
    # Initialize configuration
    config = default_config

    # Print header
    print_section_header("作业：SVD矩阵分解与降维")

    # Step 1: Generate sparse matrix
    print_section_header("【步骤1】构造稀疏矩阵", "-")

    generator = SparseMatrixGenerator(config)
    matrix_A = generator.generate_matrix()
    generator.print_matrix_info(matrix_A)

    # Step 2a: SVD decomposition and dimensionality reduction
    print_section_header("【步骤2(a)】SVD分解与降维", "-")

    svd_processor = SVDProcessor(config)
    svd_result = svd_processor.decompose(matrix_A)

    svd_processor.print_svd_info(svd_result)
    svd_processor.print_reconstruction_metrics(matrix_A, svd_result)

    # Step 2b: Find non-intersecting rows and compute similarity
    print_section_header("【步骤2(b)】选择两行并计算低维相似度", "-")

    analyzer = SimilarityAnalyzer(config)
    row_i, row_j, nonzero_i, nonzero_j = analyzer.find_non_intersecting_rows(matrix_A)

    if row_i is not None and row_j is not None:
        # Analyze similarity in low-dimensional space
        similarity_result = analyzer.analyze_row_similarity(
            original_matrix=matrix_A,
            lowdim_matrix=svd_result.U_r,
            row_i=row_i,
            row_j=row_j,
            nonzero_i=nonzero_i,
            nonzero_j=nonzero_j
        )

        analyzer.print_similarity_analysis(matrix_A, similarity_result)
    else:
        print("未找到满足条件的两行（非零元素位置完全不相交）")
        print("这在稀疏矩阵中可能发生，可以尝试增加密度或调整随机种子")

    # Print footer
    print_section_header("作业完成！")


if __name__ == "__main__":
    main()
