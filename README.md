# BigData

## 环境要求
Python 3.7+

## 目录
- [SVD矩阵分解与降维作业](#SVD矩阵分解与降维作业)
    - [作业要求](#-作业要求)
    - [算法原理](#-算法原理)
    - [参数调整](#-参数调整)
- [混合推荐系统作业](#混合推荐系统作业)
    - [作业要求](#-作业要求-1)
    - [算法架构](#-算法架构)
    - [使用方法](#-使用方法)
- [高维数据可视化分析作业](#高维数据可视化分析作业)
    - [作业要求](#-作业要求-2)
    - [可视化技术](#-可视化技术)
    - [使用方法](#-使用方法-1)

## [SVD矩阵分解与降维作业](./1-svd)
实现了基于奇异值分解（SVD）的稀疏矩阵降维分析，模拟推荐系统中的用户-商家评分矩阵处理。通过SVD分解和降维，展示了如何在低维空间中发现用户之间的潜在相似性。

### 📋 作业要求
#### 1. 稀疏矩阵构造
随机构造一个 100×1000 的用户-商家稀疏矩阵 A，确保矩阵满足稀疏性特征。

#### 2. SVD分解与降维
**任务(a)**:
- 利用SVD对矩阵A进行分解，求解U和V矩阵
- 在 r=10 的条件下计算降维矩阵

**任务(b)**:
- 从矩阵A中选择两行 i 和 j，使得 A_i 和 A_j 中非零元素的交集为空
- 计算 i 和 j 在低维空间的相似度

#### 🔧 技术实现

##### 核心算法
- **SVD分解**: A ≈ U @ Σ @ V^T
    - U: 用户的潜在特征矩阵
    - Σ: 奇异值对角矩阵
    - V^T: 商家的潜在特征矩阵

##### 关键技术点
1. **稀疏矩阵生成**: 使用 `scipy.sparse.random` 生成密度为5%的稀疏矩阵
2. **SVD分解**: 使用 `scipy.linalg.svd` 进行完整分解
3. **降维处理**: 保留前r=10个最大奇异值
4. **相似度计算**: 使用余弦相似度衡量低维空间中的用户相似性

#### 📊 预期输出

程序将输出以下内容：

##### 1. 稀疏矩阵信息
```
矩阵形状: (100, 1000)
非零元素数量: 5000
稀疏度: 95.00%
密度: 5.00%
```

##### 2. SVD分解结果
```
U矩阵形状: (100, 100)
奇异值向量形状: (100,)
V^T矩阵形状: (100, 1000)
前20个奇异值: [...]
```

##### 3. 降维效果
```
降维后的矩阵维度:
U_r: (100, 10)
Sigma_r: (10, 10)
Vt_r: (10, 1000)
重构误差: ...
能量保留率: ...%
```

##### 4. 相似度分析
```
找到的两行: 行X 和 行Y
余弦相似度: ...
欧氏距离: ...
```

### 🧮 算法原理

#### 为什么要用SVD？
SVD是一种强大的矩阵分解技术，在推荐系统中有重要应用：

1. **降维**: 从高维稀疏空间映射到低维稠密空间
2. **去噪**: 去除数据中的噪声，提取主要模式
3. **协同过滤**: 发现用户和商家之间的潜在关联

#### 稀疏性处理
在真实推荐系统中：
- 用户只评价了少数商家（稀疏性）
- 大部分用户-商家组合是未知的
- SVD可以通过潜在因子填补这些空白

#### 相似度计算意义
即使两个用户没有评价过相同的商家（原始空间无交集），通过SVD降维后，仍可能在低维空间中发现相似性。这是因为SVD捕捉了**潜在的偏好模式**。

**例如**:
- 用户A评价了{餐厅1, 餐厅2, 餐厅3}
- 用户B评价了{餐厅4, 餐厅5, 餐厅6}
- 虽然没有交集，但如果两组餐厅都属于"高档法餐"，SVD会发现A和B有相似的"高档法餐偏好"因子

### 🛠️ 参数调整

可以修改以下参数来观察不同效果：

```python
# 矩阵大小
n_users = 100    # 用户数
n_items = 1000   # 商家数

# 稀疏度
density = 0.05   # 5%的元素非零

# 降维维度
r = 10           # 保留的奇异值数量

# 评分范围
# data_rvs=lambda s: np.random.randint(1, 6, size=s)  # 1-5分
```

---

## [混合推荐系统作业](./2-recommendation_system)

实现了混合推荐系统，结合**基线模型（Baseline）**、**矩阵分解（Matrix Factorization）**和**协同过滤（Collaborative Filtering）**三种方法，将RMSE从0.89优化到0.85以下。该实现基于Netflix Prize获奖方案的核心思想。

### 📋 作业要求

#### 目标
优化推荐系统性能，使测试集RMSE < 0.85

#### 数据集
- 用户数：600
- 物品数：200
- 稀疏度：~95%（密度5%）
- 评分范围：1-5分

#### 性能指标
- **训练集RMSE**: 反映模型拟合能力
- **测试集RMSE**: 反映模型泛化能力
- **目标**: 测试集RMSE < 0.85

### 🏗️ 算法架构

#### 三层混合建模

```
第一层: Baseline Model (全局效应)
├─ 全局平均评分 μ
├─ 用户偏置 b_u（评分习惯）
└─ 物品偏置 b_i（物品质量）
预测: b_ui = μ + b_u + b_i

第二层: Matrix Factorization (区域效应)
├─ 用户隐因子矩阵 P (m×k)
└─ 物品隐因子矩阵 Q (n×k)
预测: mf_ui = q_i^T · p_u

第三层: Collaborative Filtering (局部效应)
├─ 物品相似度计算（Pearson相关）
├─ K近邻选择
└─ 学习权重矩阵 W
预测: cf_ui = Σ w_ij(r_uj - b_uj)

最终预测: r̂_ui = b_ui + α_mf·mf_ui + α_cf·cf_ui
```

#### 核心技术

1. **基线建模**
   - 捕获整体评分趋势
   - 用户评分习惯（严格/宽松）
   - 物品质量差异

2. **矩阵分解（SVD）**
   - 隐因子维度：300
   - SGD优化算法
   - L2正则化防过拟合
   - 学习率自适应衰减

3. **协同过滤**
   - 基于物品的CF
   - Pearson相似度计算
   - Top-K邻居（K=30）
   - 权重矩阵学习

4. **模型集成**
   - 加权融合三层预测
   - α_mf = 0.7（MF权重）
   - α_cf = 0.3（CF权重）

### 🚀 使用方法

#### 1. 安装依赖

```bash
cd 2-recommendation_system
pip install -r requirements.txt
```

#### 2. 运行实验

```bash
python run_experiment.py
```

输出示例：
```
==========================================
混合推荐系统实验
Hybrid Recommender System Experiment
==========================================

Generating Synthetic Data...
Matrix shape: (600, 200)
Total ratings: 18234
Density: 15.20%

Training Model...
Step 1: Computing Baseline Model...
Step 2: Initializing Latent Factors...
Step 3: Training Matrix Factorization...
  Epoch 20/120, RMSE: 0.8654
  Epoch 40/120, RMSE: 0.8432
  ...

📊 Performance Metrics:
  ├─ Training RMSE:   0.7823
  ├─ Testing RMSE:    0.8421
  └─ Improvement:     0.0479 (baseline: 0.89)

🎯 Target Achievement:
  ✅ SUCCESS! RMSE (0.8421) < 0.85
  🎉 Target achieved with margin: 0.0079
```

#### 3. 可视化结果

```bash
python visualize.py
```

生成图表：
- `training_curve.png` - 训练曲线
- `performance_comparison.png` - 性能对比
- `convergence_analysis.png` - 收敛分析
- `summary_dashboard.png` - 综合仪表板

#### 4. 查看算法原理

详细的伪代码和数学推导请参考：[algorithm_explanation.md](./2-recommendation_system/algorithm_explanation.md)

### 📊 预期输出

#### 文件结构
```
2-recommendation_system/
├── recommender.py              # 核心算法实现
├── run_experiment.py           # 实验主程序
├── visualize.py                # 可视化脚本
├── algorithm_explanation.md    # 算法原理与伪代码
├── requirements.txt            # 依赖包
├── experiment_results.json     # 实验结果（运行后生成）
└── *.png                       # 可视化图表（运行后生成）
```

#### 性能指标
- **测试集RMSE**: 0.84左右（< 0.85 ✓）
- **训练时间**: 约1-2分钟（取决于硬件）
- **改进幅度**: 相比基线（0.89）提升约5.6%

### 🔧 超参数配置

核心超参数（可在`run_experiment.py`中调整）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_factors` | 300 | 隐因子维度（越大表达能力越强） |
| `n_epochs` | 120 | 训练轮数 |
| `lr` | 0.007 | 学习率 |
| `reg_user` | 0.04 | 用户正则化系数 |
| `reg_item` | 0.04 | 物品正则化系数 |
| `k_neighbors` | 30 | 协同过滤邻居数 |
| `alpha_mf` | 0.7 | 矩阵分解权重 |
| `alpha_cf` | 0.3 | 协同过滤权重 |

### 🧮 算法原理

#### 最终预测公式

```
r̂_ui = μ + b_u + b_i + q_i^T·p_u + Σ_{j∈N(i;u)} w_ij(r_uj - b_uj)
```

其中：
- `μ`: 全局平均评分
- `b_u`, `b_i`: 用户/物品偏置
- `q_i`, `p_u`: 隐因子向量（维度k=300）
- `w_ij`: 学习到的物品权重
- `N(i;u)`: 用户u评分过的、与物品i最相似的K个物品

#### 优化目标

```
minimize Σ(r_ui - r̂_ui)² + λ_u·||P||² + λ_i·||Q||²
```

使用随机梯度下降（SGD）进行优化。

---

## [高维数据可视化分析作业](./3-visualization_analysis)

基于MNIST手写数字识别数据集，实现了9种不同的可视化分析技术，全面探索高维图像数据的特征分布、降维效果、相似性和分类性能。项目包含Python数据分析和自动化Word报告生成。

### 📋 作业要求

#### 1. 数据集要求
- **样本数量**: ≥ 1,000（实际：70,000）✅
- **特征维度**: ≥ 50（实际：784）✅
- **可视化技术**: ≥ 3种（实际：9种）✅

#### 2. 数据集信息
- **来源**: [Kaggle MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **训练集**: 60,000个样本
- **测试集**: 10,000个样本
- **图像尺寸**: 28×28像素 = 784特征
- **类别数**: 10个数字（0-9）
- **格式**: CSV文件，像素值范围0-255

#### 3. 分析任务
通过多种可视化技术回答以下核心问题：
1. **特征分布**: 像素空间的分布特征？哪些像素最重要？
2. **类别可分性**: 不同数字的相似度？容易混淆的数字对？
3. **降维效果**: 能用更少特征表示数字吗？信息保留率？
4. **分类难点**: 机器学习模型面临的挑战？

### 🎨 可视化技术

项目实现了9种互补的可视化方法：

#### 1. 数字样本展示 (Digit Sample Gallery)
- **目的**: 展示每个数字的代表性样本
- **方法**: 10×10网格布局，每个数字10个样本
- **输出**: `01_digit_samples.png`
- **发现**: 数字1、7结构简单，数字4、9变化大

#### 2. 像素强度分布 (Pixel Distribution)
- **目的**: 分析像素值统计特征
- **方法**: 直方图 + 平均数字热力图
- **输出**: `02_pixel_distribution.png`
- **发现**: 双峰分布（背景0值 vs 前景高值）

#### 3. 像素重要性热力图 (Pixel Importance)
- **目的**: 识别信息量最大的像素区域
- **方法**: 方差分析 + 均值强度映射
- **输出**: `03_pixel_importance.png`
- **发现**: 边缘4-6像素可裁剪，中心区域最关键

#### 4. PCA降维分析 (PCA Analysis)
- **目的**: 线性降维效果评估
- **方法**: 主成分分析，方差解释率曲线
- **输出**: `04_pca_analysis.png`
- **发现**: 154维保留95%方差，可压缩80%

#### 5. t-SNE可视化 (t-SNE Visualization)
- **目的**: 非线性降维聚类展示
- **方法**: t-分布随机邻域嵌入
- **输出**: `05_tsne_visualization.png`
- **发现**: 数字1紧密聚集，4和9分散重叠

#### 6. 数字相似度矩阵 (Digit Similarity)
- **目的**: 量化数字间相似程度
- **方法**: 余弦相似度计算
- **输出**: `06_digit_similarity.png`
- **发现**: 数字4和9最相似（0.917），易混淆

#### 7. 随机森林特征重要性 (Feature Importance)
- **目的**: 识别分类关键像素
- **方法**: 随机森林特征重要性排序
- **输出**: `07_feature_importance.png`
- **发现**: 中心12×12区域最具区分力

#### 8. 混淆矩阵分析 (Confusion Matrix)
- **目的**: 理解分类错误模式
- **方法**: 随机森林预测混淆矩阵
- **输出**: `08_confusion_matrix.png`
- **发现**: 训练集100%准确率，识别易混淆对

#### 9. K-means聚类 (Clustering Analysis)
- **目的**: 无监督学习效果评估
- **方法**: K-means聚类（K=10）
- **输出**: `09_clustering_analysis.png`
- **发现**: 数字0、1聚类清晰，4、7、9需监督

### 🚀 使用方法

#### 快速开始（推荐）

```bash
cd 3-visualization_analysis
chmod +x setup.sh
./setup.sh
```

自动化脚本会完成：
- 环境检查（Python、Node.js）
- 解压数据集（从 `mnist_data/archive.zip`）
- 安装所有依赖
- 创建输出目录

#### 手动配置

**步骤1 - 准备数据集**:

如果已有 `archive.zip`：
```bash
cd 3-visualization_analysis
unzip mnist_data/archive.zip -d mnist_data/
```

或从Kaggle下载：
```bash
mkdir -p mnist_data
# 访问 https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
# 下载并放置到 mnist_data/ 目录
```

**步骤2 - 安装依赖**:

```bash
# Python依赖
pip install -r requirements.txt

# Node.js依赖
npm install
```

**步骤3 - 运行分析**:

**方式A: 一键完成**
```bash
npm run full
```

**方式B: 分步执行**

步骤1 - 生成可视化图表：
```bash
python3 data_analysis_and_visualization.py
```

步骤2 - 生成Word报告：
```bash
node generate_mnist_report.js
```

**注意**: 完整分析约需5-10分钟，t-SNE最耗时（2-3分钟）

### 📊 输出结果

#### 1. 可视化图表
位置: `mnist_visualizations/`
- 9张高分辨率PNG图表（300 DPI）
- 适合直接用于报告和演示

#### 2. 自动化报告
位置: `outputs/MNIST手写数字识别可视化分析报告.docx`

报告内容：
- **封面页**: 项目标题、日期信息
- **目录**: 自动生成，带页码跳转
- **第1章**: 引言（研究背景、数据集来源）
- **第2章**: 问题定义与研究目标
- **第3章**: 数据处理过程
- **第4章**: 可视化分析与呈现（9种技术详解）
- **第5章**: 分析结论与优化建议
- **第6章**: 总结与展望

### 🔬 核心发现

#### 数据特征
- **特征分布**: 像素值双峰分布，边缘4-6像素信息量低
- **降维潜力**: 154维可保留95%方差（降维80%）
- **非线性结构**: PCA效果有限，t-SNE揭示更清晰聚类

#### 类别特征
- **易识别**: 数字0、1结构简单，聚类紧密
- **易混淆**: 数字4↔9（相似度0.917），3↔5，7↔1
- **变异大**: 数字4、7、9书写风格差异大，分布分散

#### 分类洞察
- **关键区域**: 中心12×12像素最重要
- **分类性能**: 随机森林训练集100%准确率
- **优化方向**: 边缘裁剪、数据增强、难例挖掘

### 🔧 技术栈

#### Python分析
- `pandas` (≥2.0.0) - 数据处理
- `numpy` (≥1.24.0) - 数值计算
- `matplotlib` (≥3.7.0) - 可视化绘图
- `scikit-learn` (≥1.3.0) - 机器学习（PCA、t-SNE、随机森林、K-means）

#### 报告生成
- `Node.js` (≥14.0) - 运行环境
- `docx` (^8.5.0) - Word文档生成库

### 📁 文件结构

```
3-visualization_analysis/
├── data_analysis_and_visualization.py    # 核心分析脚本
├── generate_mnist_report.js              # 报告生成脚本
├── requirements.txt                       # Python依赖
├── package.json                          # Node.js依赖
├── README.md                             # 项目文档
├── mnist_data/                           # 数据集目录（需自行下载）
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── mnist_visualizations/                 # 可视化输出
│   ├── 01_digit_samples.png
│   ├── 02_pixel_distribution.png
│   ├── 03_pixel_importance.png
│   ├── 04_pca_analysis.png
│   ├── 05_tsne_visualization.png
│   ├── 06_digit_similarity.png
│   ├── 07_feature_importance.png
│   ├── 08_confusion_matrix.png
│   └── 09_clustering_analysis.png
└── outputs/                              # 报告输出
    └── MNIST手写数字识别可视化分析报告.docx
```

### 🛠️ 参数配置

可在 `data_analysis_and_visualization.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLE_SIZE` | 10,000 | 分析样本数（最大60,000） |
| `tsne_sample_size` | 3,000 | t-SNE样本数（影响速度） |
| `n_components` | 300 | PCA主成分数量 |
| `n_estimators` | 100 | 随机森林树的数量 |
| `n_clusters` | 10 | K-means聚类数 |

### 🎯 学术价值

该项目展示了：
1. **多维度分析**: 从统计、降维、相似性、分类多角度探索数据
2. **方法论**: 如何系统性地分析高维图像数据集
3. **可重现性**: 完整的代码、文档和自动化报告
4. **实用性**: 为模型选择、特征工程提供数据依据

适用于数据可视化、机器学习、计算机视觉等课程的课程设计和实验报告。

