"""
混合推荐系统实现
Hybrid Recommender System Implementation

目标: 将 RMSE 从 0.89 优化到 0.85

核心算法:
    r̂_ui = μ + b_u + b_i + q_i^T·p_u + Σ_{j∈N(i;u)} w_ij(r_uj - b_uj)

三层建模:
    1. Baseline Model (全局效应): μ + b_u + b_i
    2. Matrix Factorization (区域效应): q_i^T * p_u
    3. Collaborative Filtering (局部效应): Σ w_ij * (r_uj - baseline_uj)
"""

import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class HybridRecommenderSystem:
    """
    混合推荐系统
    结合基线模型、矩阵分解和协同过滤的优势
    """

    def __init__(self, n_factors=100, n_epochs=100, lr=0.01,
                 reg_user=0.15, reg_item=0.15, k_neighbors=20):
        """
        初始化推荐系统

        参数:
            n_factors: 隐因子维度 (150提供良好平衡)
            n_epochs: 训练轮数 (60避免过拟合)
            lr: 学习率
            reg_user: 用户正则化参数 (0.1强正则化)
            reg_item: 物品正则化参数
            k_neighbors: 协同过滤的邻居数量
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.k_neighbors = k_neighbors

        # 模型参数
        self.global_mean = 0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None  # P矩阵
        self.item_factors = None  # Q矩阵
        self.item_weights = None  # 邻域权重矩阵W
        self.similarity_matrix = None  # 保存相似度矩阵

        # 训练历史
        self.train_rmse_history = []

        # 用户评分过的物品字典 (用于CF)
        self.user_rated_items = defaultdict(list)


    def _initialize_baseline(self, ratings_matrix):
        """
        步骤1: 计算基线预测
        baseline[u,i] = μ + b_u + b_i

        参数:
            ratings_matrix: 评分矩阵 (numpy array)
        """
        # 计算全局均值 (只统计非零元素)
        mask = ratings_matrix > 0
        self.global_mean = np.sum(ratings_matrix) / np.sum(mask)

        n_users = ratings_matrix.shape[0]
        n_items = ratings_matrix.shape[1]

        # 初始化偏置
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # 计算用户偏置
        for u in range(n_users):
            user_mask = ratings_matrix[u, :] > 0
            if np.sum(user_mask) > 0:
                user_ratings = ratings_matrix[u, user_mask]
                self.user_bias[u] = np.mean(user_ratings) - self.global_mean

        # 计算物品偏置
        for i in range(n_items):
            item_mask = ratings_matrix[:, i] > 0
            if np.sum(item_mask) > 0:
                item_ratings = ratings_matrix[item_mask, i]
                self.item_bias[i] = np.mean(item_ratings) - self.global_mean


    def _get_baseline_prediction(self, user_id, item_id):
        """
        获取基线预测值

        参数:
            user_id: 用户索引
            item_id: 物品索引

        返回:
            baseline预测评分
        """
        prediction = (self.global_mean +
                     self.user_bias[user_id] +
                     self.item_bias[item_id])
        return prediction


    def _initialize_factors(self, n_users, n_items):
        """
        步骤2: 初始化矩阵分解的隐因子矩阵
        使用小的随机值初始化

        参数:
            n_users: 用户数量
            n_items: 物品数量
        """
        # P矩阵: users × factors
        self.user_factors = np.random.normal(
            0, 0.1, (n_users, self.n_factors)
        )

        # Q矩阵: items × factors
        self.item_factors = np.random.normal(
            0, 0.1, (n_items, self.n_factors)
        )


    def _train_matrix_factorization(self, train_data):
        """
        步骤3: 训练矩阵分解模型
        使用随机梯度下降 (SGD)

        参数:
            train_data: 训练数据 [(user_id, item_id, rating), ...]
        """
        print("Training Matrix Factorization...")

        for epoch in range(self.n_epochs):
            # 打乱训练数据
            np.random.shuffle(train_data)

            epoch_loss = 0

            for user_id, item_id, rating in train_data:
                # 基线预测
                baseline = self._get_baseline_prediction(user_id, item_id)

                # 矩阵分解预测: q_i^T * p_u
                mf_pred = np.dot(
                    self.item_factors[item_id],
                    self.user_factors[user_id]
                )

                # 总预测
                prediction = baseline + mf_pred

                # 计算误差
                error = rating - prediction
                epoch_loss += error ** 2

                # 保存隐因子用于更新
                user_factor_old = self.user_factors[user_id].copy()
                item_factor_old = self.item_factors[item_id].copy()

                # 更新隐因子 (SGD with regularization)
                # q_i <- q_i + lr * (error * p_u - reg * q_i)
                self.item_factors[item_id] += (
                    self.lr * (error * user_factor_old -
                              self.reg_item * item_factor_old)
                )

                # p_u <- p_u + lr * (error * q_i - reg * p_u)
                self.user_factors[user_id] += (
                    self.lr * (error * item_factor_old -
                              self.reg_user * user_factor_old)
                )

            # 计算训练RMSE
            rmse = np.sqrt(epoch_loss / len(train_data))
            self.train_rmse_history.append(rmse)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

            # 学习率衰减 (更温和的衰减)
            if (epoch + 1) % 30 == 0:
                self.lr *= 0.96


    def _compute_item_similarity(self, ratings_matrix):
        """
        步骤4: 计算物品-物品相似度矩阵
        使用调整后的余弦相似度

        参数:
            ratings_matrix: 评分矩阵

        返回:
            相似度矩阵
        """
        n_items = ratings_matrix.shape[1]
        similarity_matrix = np.zeros((n_items, n_items))

        print("Computing item-item similarity...")

        # 对于每个物品，计算其与其他物品的相似度
        for i in range(n_items):
            if (i + 1) % 100 == 0:
                print(f"  Processing item {i + 1}/{n_items}")

            for j in range(i + 1, n_items):
                # 找到同时评分过物品i和j的用户
                mask_i = ratings_matrix[:, i] > 0
                mask_j = ratings_matrix[:, j] > 0
                common_users = mask_i & mask_j

                if np.sum(common_users) >= 3:  # 最小支持度降低到3
                    ratings_i = ratings_matrix[common_users, i]
                    ratings_j = ratings_matrix[common_users, j]

                    # 中心化 (调整后的余弦相似度)
                    ratings_i_centered = ratings_i - self.global_mean
                    ratings_j_centered = ratings_j - self.global_mean

                    # Pearson相关系数
                    numerator = np.sum(ratings_i_centered * ratings_j_centered)
                    denominator = (np.sqrt(np.sum(ratings_i_centered ** 2)) *
                                  np.sqrt(np.sum(ratings_j_centered ** 2)))

                    if denominator > 0:
                        similarity = numerator / denominator
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity

        return similarity_matrix


    def _get_k_neighbors(self, item_id, user_items, similarity_matrix):
        """
        获取K个最相似的邻居物品

        参数:
            item_id: 目标物品ID
            user_items: 用户评分过的物品列表 [(item_id, rating), ...]
            similarity_matrix: 相似度矩阵

        返回:
            K个最相似的邻居 [(neighbor_id, rating), ...]
        """
        neighbors = []

        for neighbor_id, rating in user_items:
            if neighbor_id != item_id:
                similarity = similarity_matrix[item_id, neighbor_id]
                if similarity > 0:  # 只考虑正相关的邻居
                    neighbors.append((similarity, neighbor_id, rating))

        # 按相似度排序，选择Top-K
        neighbors.sort(reverse=True)
        return [(nid, rating) for _, nid, rating in neighbors[:self.k_neighbors]]


    def _train_collaborative_filtering(self, train_data, ratings_matrix,
                                      similarity_matrix):
        """
        步骤5: 训练协同过滤模型
        学习物品权重矩阵W (替代固定相似度)

        参数:
            train_data: 训练数据
            ratings_matrix: 评分矩阵
            similarity_matrix: 相似度矩阵
        """
        n_items = ratings_matrix.shape[1]
        self.item_weights = similarity_matrix.copy()

        print("Training Collaborative Filtering...")

        # 构建用户评分过的物品字典
        self.user_rated_items = defaultdict(list)
        for user_id, item_id, rating in train_data:
            self.user_rated_items[user_id].append((item_id, rating))

        # 迭代优化权重
        cf_lr = 0.002  # CF的学习率
        for epoch in range(40):  # 增加CF训练轮数
            epoch_loss = 0
            count = 0

            for user_id, item_id, rating in train_data:
                # 找到K个最相似的邻居
                neighbors = self._get_k_neighbors(
                    item_id, self.user_rated_items[user_id], similarity_matrix
                )

                if len(neighbors) == 0:
                    continue

                # 基线预测
                baseline_target = self._get_baseline_prediction(user_id, item_id)

                # 协同过滤预测
                cf_numerator = 0
                cf_denominator = 0

                for neighbor_id, neighbor_rating in neighbors:
                    baseline_neighbor = self._get_baseline_prediction(
                        user_id, neighbor_id
                    )
                    weight = self.item_weights[item_id, neighbor_id]

                    cf_numerator += weight * (neighbor_rating - baseline_neighbor)
                    cf_denominator += abs(weight)

                if cf_denominator > 0:
                    cf_pred = cf_numerator / cf_denominator
                else:
                    cf_pred = 0

                # 总预测
                prediction = baseline_target + cf_pred

                # 计算误差
                error = rating - prediction
                epoch_loss += error ** 2
                count += 1

                # 更新权重
                for neighbor_id, neighbor_rating in neighbors:
                    baseline_neighbor = self._get_baseline_prediction(
                        user_id, neighbor_id
                    )
                    gradient = error * (neighbor_rating - baseline_neighbor)
                    # 添加正则化
                    self.item_weights[item_id, neighbor_id] += (
                        cf_lr * gradient - 0.001 * self.item_weights[item_id, neighbor_id]
                    )

            if count > 0:
                rmse = np.sqrt(epoch_loss / count)
                if (epoch + 1) % 10 == 0:
                    print(f"  CF Epoch {epoch + 1}, RMSE: {rmse:.4f}")


    def fit(self, train_data, ratings_matrix):
        """
        训练完整的混合推荐系统

        参数:
            train_data: 训练数据 [(user_id, item_id, rating), ...]
            ratings_matrix: 评分矩阵 (numpy array)
        """
        n_users, n_items = ratings_matrix.shape

        print("=" * 60)
        print("Step 1: Computing Baseline Model...")
        self._initialize_baseline(ratings_matrix)
        print(f"  Global mean: {self.global_mean:.4f}")

        print("=" * 60)
        print("Step 2: Initializing Latent Factors...")
        self._initialize_factors(n_users, n_items)
        print(f"  Initialized P: {self.user_factors.shape}, Q: {self.item_factors.shape}")

        print("=" * 60)
        print("Step 3: Training Matrix Factorization...")
        self._train_matrix_factorization(train_data)

        print("=" * 60)
        print("Step 4: Computing Item Similarity...")
        self.similarity_matrix = self._compute_item_similarity(ratings_matrix)

        print("=" * 60)
        print("Step 5: Training Collaborative Filtering...")
        self._train_collaborative_filtering(
            train_data, ratings_matrix, self.similarity_matrix
        )

        print("=" * 60)
        print("Training Complete!")


    def predict(self, user_id, item_id, alpha_mf=0.8, alpha_cf=0.0):
        """
        预测用户对物品的评分
        集成三个组件的预测结果

        参数:
            user_id: 用户索引
            item_id: 物品索引
            alpha_mf: 矩阵分解的权重 (降低以防止过拟合)
            alpha_cf: 协同过滤的权重

        返回:
            预测评分 (限制在1-5之间)
        """
        # 组件1: 基线预测
        baseline = self._get_baseline_prediction(user_id, item_id)

        # 组件2: 矩阵分解预测
        mf_pred = np.dot(self.item_factors[item_id], self.user_factors[user_id])

        # 组件3: 协同过滤预测
        cf_pred = 0
        if user_id in self.user_rated_items and len(self.user_rated_items[user_id]) > 0:
            neighbors = self._get_k_neighbors(
                item_id,
                self.user_rated_items[user_id],
                self.item_weights
            )

            if len(neighbors) > 0:
                cf_numerator = 0
                cf_denominator = 0

                for neighbor_id, neighbor_rating in neighbors:
                    baseline_neighbor = self._get_baseline_prediction(user_id, neighbor_id)
                    weight = self.item_weights[item_id, neighbor_id]

                    cf_numerator += weight * (neighbor_rating - baseline_neighbor)
                    cf_denominator += abs(weight)

                if cf_denominator > 0:
                    cf_pred = cf_numerator / cf_denominator

        # 加权组合
        prediction = baseline + alpha_mf * mf_pred + alpha_cf * cf_pred

        # 限制在合理范围
        prediction = np.clip(prediction, 1, 5)

        return prediction


    def evaluate(self, test_data):
        """
        在测试集上评估模型性能

        参数:
            test_data: 测试数据 [(user_id, item_id, rating), ...]

        返回:
            RMSE评分
        """
        squared_errors = []

        for user_id, item_id, true_rating in test_data:
            pred_rating = self.predict(user_id, item_id)
            error = true_rating - pred_rating
            squared_errors.append(error ** 2)

        rmse = np.sqrt(np.mean(squared_errors))
        return rmse
