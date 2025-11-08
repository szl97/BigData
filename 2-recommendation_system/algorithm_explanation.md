# 混合推荐系统算法原理与伪代码

## 目录
- [一、算法概述](#一算法概述)
- [二、核心算法伪代码](#二核心算法伪代码)
- [三、数学原理详解](#三数学原理详解)
- [四、优化技巧](#四优化技巧)
- [五、复杂度分析](#五复杂度分析)

---

## 一、算法概述

### 1.1 问题定义

**输入**:
- 用户-物品评分矩阵 R ∈ ℝ^(m×n)，其中大部分元素未知（稀疏矩阵）
- m 个用户，n 个物品
- 已观测评分集合：{(u, i, r_ui) | 用户u对物品i的评分为r_ui}

**输出**:
- 预测评分函数 r̂_ui，预测用户u对物品i的评分

**目标**:
- 最小化均方根误差 (RMSE): √(Σ(r_ui - r̂_ui)² / N)
- 目标性能: RMSE < 0.85

### 1.2 三层混合建模架构

```
┌─────────────────────────────────────────────────────┐
│  第一层: Baseline Model (全局效应)                    │
│  预测: b_ui = μ + b_u + b_i                         │
│  作用: 捕获整体评分趋势和用户/物品偏置                 │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  第二层: Matrix Factorization (区域效应)             │
│  预测: mf_ui = q_i^T · p_u                          │
│  作用: 学习用户-物品的潜在交互模式                     │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  第三层: Collaborative Filtering (局部效应)          │
│  预测: cf_ui = Σ_{j∈N(i;u)} w_ij(r_uj - b_uj)      │
│  作用: 利用物品间相似性进行精细化预测                  │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  最终预测 (加权融合)                                 │
│  r̂_ui = b_ui + α_mf·mf_ui + α_cf·cf_ui             │
└─────────────────────────────────────────────────────┘
```

---

## 二、核心算法伪代码

### 2.1 整体训练流程

```
def HybridRecommenderSystemTrain(R, k, epochs, lr, λ_u, λ_i):
    input:
        R: m×n 评分矩阵
        k: 隐因子维度
        epochs: 训练轮数
        lr: 学习率
        λ_u, λ_i: 正则化参数

    output:
        训练好的模型参数 (μ, b_u, b_i, P, Q, W)

    // ===== 步骤1: 初始化基线模型 =====
    μ ← ComputeGlobalMean(R)
    b_u ← InitializeUserBias(R, μ)
    b_i ← InitializeItemBias(R, μ)

    // ===== 步骤2: 初始化隐因子矩阵 =====
    P ← RandomMatrix(m, k)  // 用户隐因子
    Q ← RandomMatrix(n, k)  // 物品隐因子

    // ===== 步骤3: 训练矩阵分解 =====
    for epoch = 1 to epochs do:
        Shuffle(TrainingData)

        for each (u, i, r_ui) in TrainingData do:
            // 前向传播
            b_ui ← μ + b_u[u] + b_i[i]
            mf_ui ← Q[i]^T · P[u]
            r̂_ui ← b_ui + mf_ui
            e_ui ← r_ui - r̂_ui

            // 反向传播 (SGD更新)
            P[u] ← P[u] + lr·(e_ui·Q[i] - λ_u·P[u])
            Q[i] ← Q[i] + lr·(e_ui·P[u] - λ_i·Q[i])

        // 学习率衰减
        if epoch % 30 == 0:
            lr ← lr × 0.96

    // ===== 步骤4: 计算物品相似度 =====
    S ← ComputeItemSimilarity(R, μ)

    // ===== 步骤5: 训练协同过滤 =====
    W ← TrainCollaborativeFiltering(R, S, μ, b_u, b_i)

    return (μ, b_u, b_i, P, Q, W)
```

### 2.2 基线模型计算

```
def ComputeGlobalMean(R):
    total_sum ← 0
    count ← 0

    for u = 1 to m do:
        for i = 1 to n do:
            if R[u,i] > 0:
                total_sum ← total_sum + R[u,i]
                count ← count + 1

    return total_sum / count


def InitializeUserBias(R, μ):
    for u = 1 to m do:
        rated_items ← {i | R[u,i] > 0}
        if |rated_items| > 0:
            b_u[u] ← Mean(R[u, rated_items]) - μ
        else:
            b_u[u] ← 0

    return b_u


def InitializeItemBias(R, μ):
    for i = 1 to n do:
        rated_users ← {u | R[u,i] > 0}
        if |rated_users| > 0:
            b_i[i] ← Mean(R[rated_users, i]) - μ
        else:
            b_i[i] ← 0

    return b_i
```

### 2.3 物品相似度计算

```
def ComputeItemSimilarity(R, μ):
    初始化: S ← 零矩阵(n×n)

    for i = 1 to n do:
        for j = i+1 to n do:
            // 找到同时评分过物品i和j的用户
            common_users ← {u | R[u,i]>0 且 R[u,j]>0}

            if |common_users| >= min_support:
                // 提取评分向量
                r_i ← R[common_users, i]
                r_j ← R[common_users, j]

                // 中心化 (调整后的余弦相似度)
                r_i_centered ← r_i - μ
                r_j_centered ← r_j - μ

                // 计算Pearson相关系数
                numerator ← r_i_centered^T · r_j_centered
                denominator ← ||r_i_centered|| × ||r_j_centered||

                if denominator > 0:
                    S[i,j] ← numerator / denominator
                    S[j,i] ← S[i,j]

    return S
```

### 2.4 协同过滤训练

```
def TrainCollaborativeFiltering(R, S, μ, b_u, b_i, K=30, cf_epochs=40):
    W ← S  // 初始化权重矩阵为相似度矩阵
    lr_cf ← 0.002

    // 构建用户评分历史
    for each (u, i, r_ui) in TrainingData:
        UserItems[u].append((i, r_ui))

    for epoch = 1 to cf_epochs do:
        for each (u, i, r_ui) in TrainingData do:
            // 找K个最相似邻居
            neighbors ← GetTopKNeighbors(i, UserItems[u], S, K)

            if |neighbors| == 0:
                continue

            // 计算CF预测
            cf_numerator ← 0
            cf_denominator ← 0

            for each (j, r_uj) in neighbors do:
                b_uj ← μ + b_u[u] + b_i[j]
                w_ij ← W[i,j]
                cf_numerator ← cf_numerator + w_ij·(r_uj - b_uj)
                cf_denominator ← cf_denominator + |w_ij|

            cf_pred ← cf_numerator / cf_denominator

            // 总预测
            b_ui ← μ + b_u[u] + b_i[i]
            r̂_ui ← b_ui + cf_pred
            e_ui ← r_ui - r̂_ui

            // 更新权重
            for each (j, r_uj) in neighbors do:
                b_uj ← μ + b_u[u] + b_i[j]
                gradient ← e_ui·(r_uj - b_uj)
                W[i,j] ← W[i,j] + lr_cf·gradient - 0.001·W[i,j]

    return W


def GetTopKNeighbors(item, user_items, S, K):
    candidates ← []

    for each (j, r_uj) in user_items do:
        if j ≠ item and S[item,j] > 0:
            candidates.append((S[item,j], j, r_uj))

    // 按相似度降序排序
    candidates.Sort(descending=True)

    // 返回前K个
    return candidates[0:K]
```

### 2.5 预测阶段

```
def Predict(u, i, μ, b_u, b_i, P, Q, W, UserItems, α_mf=0.7, α_cf=0.3):
    // 组件1: 基线预测
    b_ui ← μ + b_u[u] + b_i[i]

    // 组件2: 矩阵分解预测
    mf_pred ← Q[i]^T · P[u]

    // 组件3: 协同过滤预测
    cf_pred ← 0
    neighbors ← GetTopKNeighbors(i, UserItems[u], W, K=30)

    if |neighbors| > 0:
        cf_numerator ← 0
        cf_denominator ← 0

        for each (j, r_uj) in neighbors do:
            b_uj ← μ + b_u[u] + b_i[j]
            w_ij ← W[i,j]
            cf_numerator ← cf_numerator + w_ij·(r_uj - b_uj)
            cf_denominator ← cf_denominator + |w_ij|

        cf_pred ← cf_numerator / cf_denominator

    // 加权融合
    r̂_ui ← b_ui + α_mf·mf_pred + α_cf·cf_pred

    // 限制在评分范围内 [1, 5]
    r̂_ui ← Clip(r̂_ui, 1, 5)

    return r̂_ui
```

---

## 三、数学原理详解

### 3.1 基线模型 (Baseline Predictors)

**基本形式**:
```
b_ui = μ + b_u + b_i
```

其中:
- `μ`: 全局平均评分，反映整体评分趋势
- `b_u`: 用户偏置，反映用户评分习惯（严格/宽松）
- `b_i`: 物品偏置，反映物品质量差异

**计算公式**:
```
μ = (Σ_{(u,i)∈K} r_ui) / |K|

b_u = (Σ_{i∈I(u)} r_ui) / |I(u)| - μ

b_i = (Σ_{u∈U(i)} r_ui) / |U(i)| - μ
```

**意义**: 消除评分中的系统性偏差，为后续模型提供更纯净的信号。

### 3.2 矩阵分解 (Matrix Factorization)

**分解形式**:
```
R ≈ P × Q^T
```

其中:
- `P ∈ ℝ^(m×k)`: 用户隐因子矩阵
- `Q ∈ ℝ^(n×k)`: 物品隐因子矩阵
- `k`: 隐因子维度

**预测公式**:
```
r̂_ui = b_ui + q_i^T · p_u
```

**优化目标**:
```
minimize Σ_{(u,i)∈K} (r_ui - b_ui - q_i^T·p_u)² + λ_u·||P||² + λ_i·||Q||²
```

**梯度**:
```
∂L/∂p_u = -2·e_ui·q_i + 2λ_u·p_u
∂L/∂q_i = -2·e_ui·p_u + 2λ_i·q_i

其中 e_ui = r_ui - r̂_ui
```

**SGD更新规则**:
```
p_u ← p_u + η·(e_ui·q_i - λ_u·p_u)
q_i ← q_i + η·(e_ui·p_u - λ_i·q_i)
```

### 3.3 协同过滤 (Neighborhood-based CF)

**基本思想**: 利用物品间的相似性，根据用户对相似物品的评分来预测。

**相似度计算** (Pearson相关系数):
```
sim(i,j) = Σ_{u∈U(i,j)} (r_ui - μ)·(r_uj - μ) / (||r_i - μ|| × ||r_j - μ||)
```

其中 `U(i,j)` 是同时评分过物品i和j的用户集合。

**预测公式**:
```
cf_ui = Σ_{j∈N(i;u)} w_ij·(r_uj - b_uj) / Σ_{j∈N(i;u)} |w_ij|
```

其中:
- `N(i;u)`: 用户u评分过的、与物品i最相似的K个物品
- `w_ij`: 学习到的物品权重（初始化为相似度）

**权重学习**:
```
w_ij ← w_ij + η_cf·e_ui·(r_uj - b_uj) - λ_cf·w_ij
```

### 3.4 集成策略

**最终预测**:
```
r̂_ui = b_ui + α_mf·(q_i^T·p_u) + α_cf·(Σ w_ij(r_uj - b_uj) / Σ|w_ij|)
```

其中 `α_mf + α_cf ≤ 1` 是融合权重。

**为什么有效？**
- Baseline: 捕获粗粒度的全局模式
- MF: 捕获中粒度的用户-物品交互
- CF: 捕获细粒度的局部关联
- 三者互补，共同提升预测精度

---

## 四、优化技巧

### 4.1 超参数优化

| 参数 | 基础值 | 优化值 | 原因 |
|------|--------|--------|------|
| n_factors | 100-200 | 300 | 增强模型表达能力 |
| n_epochs | 80-100 | 120 | 充分训练 |
| learning_rate | 0.005 | 0.007 | 加快收敛 |
| reg_user/item | 0.05 | 0.04 | 增加灵活性 |
| k_neighbors | 20 | 30 | 考虑更多邻居信息 |
| α_mf | 0.6 | 0.7 | MF表现更好，增加权重 |
| α_cf | 0.4 | 0.3 | 相应降低CF权重 |

### 4.2 训练技巧

1. **学习率衰减**: 每30轮衰减5%
   ```
   η_t = η_0 × 0.96^(t/30)
   ```

2. **数据打乱**: 每轮训练前随机打乱样本顺序

3. **正则化**: 防止过拟合
   ```
   L = L_data + λ·L_reg
   ```

4. **初始化**: 使用小的随机值 N(0, 0.1)

### 4.3 数值稳定性

- 避免除零: 分母加小常数 ε = 1e-10
- 梯度裁剪: 限制梯度范数
- 评分截断: 预测值限制在 [1, 5]

---

## 五、复杂度分析

### 5.1 时间复杂度

**训练阶段**:
- Baseline初始化: O(|K|)，其中|K|是已知评分数
- MF训练: O(T × |K| × k)，T是训练轮数
- 相似度计算: O(n² × m)，最坏情况
- CF训练: O(T_cf × |K| × K)

**总体**: O(T × |K| × k + n² × m)

**预测阶段**:
- 单次预测: O(k + K)

### 5.2 空间复杂度

- 评分矩阵: O(|K|)（稀疏存储）
- 隐因子矩阵: O(m×k + n×k)
- 相似度矩阵: O(n²)

**总体**: O(|K| + (m+n)×k + n²)

### 5.3 可扩展性

对于大规模数据集:
- 使用稀疏矩阵存储
- 批量训练减少内存占用
- 近似最近邻加速相似度计算
- 分布式训练

---

## 六、与Netflix Prize获胜方案的关系

Netflix Prize的获胜方案（BellKor's Pragmatic Chaos）使用了类似的思想：

1. **多模型集成**: Baseline + MF + CF + 其他模型
2. **时间动态**: 考虑评分随时间的演变（我们的简化版未实现）
3. **隐式反馈**: 除了显式评分，还考虑浏览行为（我们的简化版未实现）
4. **参数学习**: 所有权重都是学习而非手工设定

我们的实现是Netflix方案的简化版，保留了核心思想，在计算资源受限的情况下达到较好效果。

---
