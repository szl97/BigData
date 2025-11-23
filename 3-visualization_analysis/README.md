# MNIST Handwritten Digit Recognition - Data Visualization Analysis

A comprehensive data visualization and analysis project for the MNIST handwritten digit dataset, implementing 9 different visualization techniques to explore data characteristics, patterns, and classification challenges.

## Project Overview

This project performs in-depth visual analysis of the MNIST dataset (70,000 handwritten digit images) using multiple data science and machine learning techniques. The analysis satisfies academic requirements with:
- **Sample size**: 10,000+ samples (far exceeding 1,000 minimum)
- **Feature dimensions**: 784 features (far exceeding 50 minimum)
- **Visualization techniques**: 9 different methods

## Dataset Information

- **Source**: [Kaggle - MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Total samples**: 70,000 images (60,000 training + 10,000 test)
- **Image size**: 28 x 28 pixels = 784 features
- **Classes**: 10 digits (0-9)
- **Format**: CSV files with pixel values (0-255)

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Node.js 14.0 or higher (for report generation)
- 2GB+ RAM (for full analysis)
- 500MB+ disk space

### Required Python Packages

Install dependencies using:

```bash
pip install -r requirements.txt
```

The project requires:
- `pandas` (>=2.0.0) - Data manipulation
- `numpy` (>=1.24.0) - Numerical computing
- `matplotlib` (>=3.7.0) - Visualization
- `scikit-learn` (>=1.3.0) - Machine learning algorithms

### Required Node.js Packages

Install dependencies using:

```bash
npm install
```

The project requires:
- `docx` (^8.5.0) - Word document generation

## Project Structure

```
3-visualization_analysis/
├── data_analysis_and_visualization.py    # Main Python analysis script
├── generate_mnist_report.js              # Report generation script
├── requirements.txt                       # Python dependencies
├── package.json                          # Node.js dependencies
├── README.md                             # This file
├── mnist_data/                           # Dataset directory (not included)
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── mnist_visualizations/                 # Visualization output (auto-created)
│   ├── 01_digit_samples.png
│   ├── 02_pixel_distribution.png
│   ├── 03_pixel_importance.png
│   ├── 04_pca_analysis.png
│   ├── 05_tsne_visualization.png
│   ├── 06_digit_similarity.png
│   ├── 07_feature_importance.png
│   ├── 08_confusion_matrix.png
│   └── 09_clustering_analysis.png
└── outputs/                              # Report output (auto-created)
    └── MNIST手写数字识别可视化分析报告.docx
```

## Setup Instructions

### Quick Start (Recommended)

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script will automatically:
- Check Python and Node.js environments
- Unzip the dataset from `mnist_data/archive.zip`
- Install all Python and Node.js dependencies
- Create output directories

### Manual Setup

#### 1. Prepare MNIST Dataset

**Option A: If you have archive.zip**

If you already have `mnist_data/archive.zip`, extract it:

```bash
unzip mnist_data/archive.zip -d mnist_data/
```

**Option B: Download from Kaggle**

Download the MNIST CSV dataset from Kaggle:

```bash
# Create data directory
mkdir -p mnist_data

# Download from Kaggle (requires Kaggle API)
# Or manually download from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
# Place archive.zip or the extracted csv files in mnist_data/
```

The dataset should contain:
- `mnist_train.csv` (105 MB, 60,000 samples)
- `mnist_test.csv` (17 MB, 10,000 samples)

#### 2. Install Dependencies

**Python dependencies:**
```bash
pip install -r requirements.txt
```

**Node.js dependencies:**
```bash
npm install
```

#### 3. Run the Complete Workflow

**Option A: Run both steps automatically**
```bash
npm run full
```

**Option B: Run steps separately**

Step 1 - Generate visualizations:
```bash
python3 data_analysis_and_visualization.py
# or use: npm run analyze
```

Step 2 - Generate Word report:
```bash
node generate_mnist_report.js
# or use: npm run generate-report
```

**Note**: The complete analysis takes approximately 5-10 minutes, with t-SNE visualization being the most time-consuming step (2-3 minutes).

## Visualization Techniques

The project implements 9 comprehensive visualization methods:

### 1. Digit Sample Gallery
- **Purpose**: Display representative samples of each digit
- **Method**: Grid visualization of 100 samples (10 per digit)
- **Output**: `01_digit_samples.png`

### 2. Pixel Intensity Distribution
- **Purpose**: Analyze pixel value distributions and average digit patterns
- **Method**: Histogram and heatmap of average images
- **Output**: `02_pixel_distribution.png`

### 3. Pixel Importance Heatmap
- **Purpose**: Identify which pixels contain most information
- **Method**: Variance analysis and mean intensity mapping
- **Output**: `03_pixel_importance.png`
- **Finding**: Center pixels have high variance (high information), edges have low variance

### 4. PCA Dimensionality Reduction
- **Purpose**: Reduce 784 dimensions while preserving variance
- **Method**: Principal Component Analysis
- **Output**: `04_pca_analysis.png`
- **Finding**: ~154 components preserve 95% variance, ~331 preserve 99%

### 5. t-SNE Visualization
- **Purpose**: Non-linear dimensionality reduction for clustering visualization
- **Method**: t-Distributed Stochastic Neighbor Embedding
- **Output**: `05_tsne_visualization.png`
- **Finding**: Reveals natural clustering structure; digit 1 is tightly clustered, 4 and 9 are dispersed

### 6. Digit Similarity Matrix
- **Purpose**: Measure inter-digit similarity
- **Method**: Cosine similarity between average digit images
- **Output**: `06_digit_similarity.png`
- **Finding**: Identifies most similar digit pairs

### 7. Random Forest Feature Importance
- **Purpose**: Identify most discriminative pixels for classification
- **Method**: Train Random Forest, analyze feature importances
- **Output**: `07_feature_importance.png`
- **Finding**: Center region pixels are most important for classification

### 8. Confusion Matrix Analysis
- **Purpose**: Understand classification errors and digit confusability
- **Method**: Random Forest predictions with confusion matrix
- **Output**: `08_confusion_matrix.png`
- **Finding**: Identifies commonly confused digit pairs (e.g., 4 vs 9, 3 vs 5)

### 9. K-means Clustering
- **Purpose**: Unsupervised clustering analysis
- **Method**: K-means with K=10 clusters
- **Output**: `09_clustering_analysis.png`
- **Finding**: Compares unsupervised clustering with true labels

## Key Findings

### Data Characteristics
- Dataset is well-balanced across all 10 digit classes
- Pixel values follow a bimodal distribution (background vs foreground)
- Edge pixels contain minimal information; center region is critical

### Dimensionality Reduction
- PCA: 154 components (19.6% of original) preserve 95% variance
- Significant compression possible without major information loss
- t-SNE reveals clearer separation than PCA for visualization

### Classification Insights
- Random Forest achieves 95%+ accuracy on training data
- Digits 1 and 0 are easiest to classify (simple, distinct patterns)
- Digits 4, 7, and 9 are more challenging (higher intra-class variance)
- Common confusions: 4↔9, 3↔5, 7↔1

### Feature Importance
- Center 12x12 pixel region contains most discriminative information
- Border pixels can be cropped with minimal accuracy loss
- Some digits (like 1) use only a small subset of features

## Performance Notes

### Sample Size Configuration
The script uses 10,000 samples by default for analysis efficiency. To use the full dataset:

```python
# In data_analysis_and_visualization.py, line 48:
SAMPLE_SIZE = 10000  # Change to 60000 for full training set
```

### t-SNE Optimization
t-SNE is computationally expensive. The script uses 3,000 samples for t-SNE. To adjust:

```python
# Line 257:
tsne_sample_size = 3000  # Increase for more samples (slower)
```

## Output

### Visualization Files
All visualizations are saved to `mnist_visualizations/` as high-resolution PNG files (300 DPI) suitable for reports and presentations.

### Word Report
The automated report generation creates a comprehensive Word document at:
```
outputs/MNIST手写数字识别可视化分析报告.docx
```

The report includes:
- **Cover page** with project title and date
- **Table of contents** with automatic page numbering
- **6 chapters**: Introduction, Problem Definition, Data Processing, Visualization Analysis, Conclusions, Summary
- **All 9 visualizations** embedded with descriptions
- **Detailed analysis** of findings and recommendations
- Professional formatting with headings, bullet points, and styled text

## Future Extensions

Potential improvements and extensions:
- Deep learning analysis (CNN feature maps)
- Data augmentation impact study
- Cross-validation for robust accuracy estimates
- Interactive visualizations with Plotly
- Real-time digit recognition demo
- Analysis of misclassified samples

## Academic Context

This project fulfills data visualization course requirements:
1. **Problem Definition**: Handwritten digit recognition data analysis
2. **Data Processing**: Normalization, standardization, dimensionality reduction
3. **Visualization**: 9 diverse techniques covering distribution, projection, similarity, and performance
4. **Analysis & Conclusions**: Identification of key features, classification challenges, and data patterns

## Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce `SAMPLE_SIZE` to 5000 or lower
- Reduce `tsne_sample_size` to 1000-2000
- Close other applications to free RAM

### Missing Dataset
```
FileNotFoundError: mnist_data/mnist_train.csv
```
Download the MNIST CSV files from Kaggle and place them in `mnist_data/` directory.

### Font Warnings
If you see font warnings, they can be safely ignored. The script uses DejaVu Sans as a fallback font.

### Report Generation Errors
If `generate_mnist_report.js` fails:
- Ensure all 9 visualization images were generated successfully in `mnist_visualizations/`
- The `outputs/` directory will be created automatically
- Check that Node.js and npm packages are properly installed

## License

This project is for educational purposes. The MNIST dataset is publicly available and free to use.

## References

- LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit database
- [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- Scikit-learn Documentation: https://scikit-learn.org/

## Contact

For questions or issues, please refer to the course materials or contact the instructor.

---

**Last Updated**: 2025-11-23
