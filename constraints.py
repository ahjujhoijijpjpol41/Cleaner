import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error
import random
from multiprocessing import Pool, cpu_count

class RowConstraintMiner:
    def __init__(self, clean_data):
        """
        初始化方法，接受一个numpy.ndarray作为输入数据
        """
        self.data = clean_data  # 输入数据为numpy数组
        self.num_rows, self.num_cols = self.data.shape  # 数据行列数

    def generate_binary_strings(self, m, k):
        """
        生成所有可能的k位二进制字符串组合，表示从m个特征中选取k个特征。
        """
        base_array = np.zeros(m, dtype=int)
        all_combinations = []

        for positions in combinations(range(m), k):
            new_array = base_array.copy()
            new_array[list(positions)] = 1
            all_combinations.append(new_array)

        return all_combinations

    def train_models_and_evaluate(self, k):
        """
        对所有二进制组合训练Ridge模型并计算损失。
        """
        m = self.data.shape[1]
        binary_strings = self.generate_binary_strings(m, k)
        models_with_loss = []

        for binary_string in binary_strings:
            selected_indices = np.where(binary_string == 1)[0]

            # 随机选择一个特征作为y
            y_index = random.choice(selected_indices)
            y = self.data[:, y_index]
            X_indices = np.delete(selected_indices, np.where(selected_indices == y_index))
            X = self.data[:, X_indices]

            # 训练 Ridge 模型
            model = Ridge()
            model.fit(X, y)
            y_pred = model.predict(X)
            loss = mean_squared_error(y, y_pred)

            models_with_loss.append((binary_string, y_index, model, loss))

        # 根据损失对模型排序
        models_with_loss.sort(key=lambda x: x[3])

        return models_with_loss

    def select_optimal_models(self, models_with_loss):
        """
        根据损失和规则选择最佳模型。
        """
        attr_selected_count = {i: 0 for i in range(self.num_cols)}  # 属性被选中的次数
        selected_models = []

        for binary_string, y_index, model, loss in models_with_loss:
            if loss > 5:  # 忽略高损失模型
                continue

            involved_indices = np.where(binary_string == 1)[0]

            overlap_count = sum([binary_string[i] for i in attr_selected_count if attr_selected_count[i] > 0])
            unique_overlap_selected_count = max(
                [attr_selected_count[i] for i in involved_indices if attr_selected_count[i] > 0], default=0
            )

            if overlap_count <= 1 and unique_overlap_selected_count < 3:
                selected_models.append((binary_string, y_index, model, loss))
                for idx in involved_indices:
                    attr_selected_count[idx] += 1

        return selected_models

    def mine_row_constraints(self, attr_num=3):
        """
        主方法：发现行约束，返回约束和覆盖的属性集合。
        """
        models_with_loss = self.train_models_and_evaluate(attr_num)
        optimal_models = self.select_optimal_models(models_with_loss)

        constraints = []
        covered_attrs = set()

        for binary_string, y_index, model, _ in optimal_models:
            selected_indices = np.where(binary_string == 1)[0]
            X_indices = np.delete(selected_indices, np.where(selected_indices == y_index))
            X = self.data[:, X_indices]
            y = self.data[:, y_index]

            # 重新训练 Ridge 模型
            new_model = Ridge()
            new_model.fit(X, y)
            y_pred = new_model.predict(X)

            # 计算每个数据点的损失
            losses = y_pred - y
            rho_min, rho_max = np.min(losses) - new_model.intercept_, np.max(losses) - new_model.intercept_

            # 准备系数数组
            full_coef = np.zeros(self.num_cols)
            full_coef[X_indices] = new_model.coef_
            full_coef[y_index] = -1

            # 更新 covered_attrs，只包括系数为 -1 的属性
            covered_attrs.add(y_index)

            # 构建约束字符串，只包括非零系数的项
            terms = [f"{coef_val:.3f} * x{i}" for i, coef_val in enumerate(full_coef) if coef_val != 0]
            constraint_str = f"{rho_min:.3f} <= {' + '.join(terms)} <= {rho_max:.3f}"

            constraints.append((constraint_str, full_coef, rho_min, rho_max))
        print('comp')
        # for constraint in constraints:
        #     print(constraints[0])
        # coding
        return constraints, covered_attrs


class RowConstraintMiner1:
    def __init__(self, clean_data):
        """
        初始化方法，接受一个numpy.ndarray作为输入数据
        """
        self.data = clean_data  # 输入数据为numpy数组
        self.num_rows, self.num_cols = self.data.shape  # 数据行列数

    def generate_binary_strings(self, m, k):
        """
        生成所有可能的k位二进制字符串组合，表示从m个特征中选取k个特征。
        """
        base_array = np.zeros(m, dtype=int)
        all_combinations = []

        for positions in combinations(range(m), k):
            new_array = base_array.copy()
            new_array[list(positions)] = 1
            all_combinations.append(new_array)

        return all_combinations

    def train_models_and_evaluate(self, k):
        """
        对所有二进制组合训练Ridge模型并计算损失。
        """
        m = self.data.shape[1]
        binary_strings = self.generate_binary_strings(m, k)
        # binary_strings = self.generate_binary_strings(self.num_cols, k)
        models_with_loss = []

        for binary_string in binary_strings:
            selected_indices = np.where(binary_string == 1)[0]

            # 随机选择一个特征作为y
            y_index = random.choice(selected_indices)
            y = self.data[:, y_index]
            X_indices = np.delete(selected_indices, np.where(selected_indices == y_index))
            X = self.data[:, X_indices]

            # 训练 Ridge 模型
            model = ElasticNet(alpha=1.0, l1_ratio=0.5)
            model.fit(X, y)
            y_pred = model.predict(X)
            loss = mean_squared_error(y, y_pred)

            models_with_loss.append((binary_string, y_index, model, loss))

        # 根据损失对模型排序
        models_with_loss.sort(key=lambda x: x[3])

        return models_with_loss

    def select_optimal_models(self, models_with_loss):
        """
        根据损失和规则选择最佳模型。
        """
        attr_selected_count = {i: 0 for i in range(self.num_cols)}  # 属性被选中的次数
        selected_models = []

        for binary_string, y_index, model, loss in models_with_loss:
            if loss > 5:  # 忽略高损失模型
                continue

            involved_indices = np.where(binary_string == 1)[0]

            overlap_count = sum([binary_string[i] for i in attr_selected_count if attr_selected_count[i] > 0])
            unique_overlap_selected_count = max(
                [attr_selected_count[i] for i in involved_indices if attr_selected_count[i] > 0], default=0
            )

            if overlap_count <= 1 and unique_overlap_selected_count < 3:
                selected_models.append((binary_string, y_index, model, loss))
                for idx in involved_indices:
                    attr_selected_count[idx] += 1

        return selected_models

    def mine_row_constraints(self, attr_num=3):
        """
        主方法：发现行约束，返回约束和覆盖的属性集合。
        """
        models_with_loss = self.train_models_and_evaluate(attr_num)
        optimal_models = self.select_optimal_models(models_with_loss)

        constraints = []
        covered_attrs = set()

        for binary_string, y_index, model, _ in optimal_models:
            selected_indices = np.where(binary_string == 1)[0]
            X_indices = np.delete(selected_indices, np.where(selected_indices == y_index))
            X = self.data[:, X_indices]
            y = self.data[:, y_index]

            # 重新训练 Ridge 模型
            new_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
            new_model.fit(X, y)
            y_pred = new_model.predict(X)

            # 计算每个数据点的损失
            losses = y_pred - y
            rho_min, rho_max = np.min(losses) - new_model.intercept_, np.max(losses) - new_model.intercept_

            # 准备系数数组
            full_coef = np.zeros(self.num_cols)
            full_coef[X_indices] = new_model.coef_
            full_coef[y_index] = -1

            # 更新 covered_attrs，只包括系数为 -1 的属性
            covered_attrs.add(y_index)

            # 构建约束字符串，只包括非零系数的项
            terms = [f"{coef_val:.3f} * x{i}" for i, coef_val in enumerate(full_coef) if coef_val != 0]
            constraint_str = f"{rho_min:.3f} <= {' + '.join(terms)} <= {rho_max:.3f}"

            constraints.append((constraint_str, full_coef, rho_min, rho_max))
        # print('comp')
        # for constraint in constraints:
        #     print(constraints[0])
        return constraints, covered_attrs

class ColConstraintMiner:
    def __init__(self, clean_data):
        self.data = clean_data

    def _calculate_speeds(self):
        speed_constraints = {}
        for col_index in range(self.data.shape[1]):  # 遍历每列
            column_data = self.data[:, col_index]
            speeds = np.diff(column_data)  # 一阶差分计算速度，保留符号
            speed_constraints[col_index] = (np.min(speeds), np.max(speeds))
        return speed_constraints

    def _calculate_accelerations(self):
        acceleration_constraints = {}
        for col_index in range(self.data.shape[1]):  # 遍历每列
            column_data = self.data[:, col_index]
            accelerations = np.diff(np.diff(column_data))  # 二阶差分计算加速度，保留符号
            acceleration_constraints[col_index] = (np.min(accelerations), np.max(accelerations))
        return acceleration_constraints

    

    def mine_col_constraints(self):
        """
        挖掘列约束，返回每列的速度和加速度约束。

        返回:
        - speed_constraints: 速度约束字典 {列索引: (最小速度, 最大速度)}
        - acceleration_constraints: 加速度约束字典 {列索引: (最小加速度, 最大加速度)}
        """
        speed_constraints = self._calculate_speeds()
        acceleration_constraints = self._calculate_accelerations()
        # variance_constraints = self._calculate_variances()
        print('comp')
        # return speed_constraints, acceleration_constraints, variance_constraints
        return speed_constraints, acceleration_constraints


class Range:
    def __init__(self, attr, l, r):
        self.attr = attr
        self.l = l
        self.r = r

class VarianceConstraint:
    def __init__(self, attr_index):
        self.attr_index = attr_index
        self.range = None
        self.hasdcv = False
        self.var_b = None
        self.var = None

    def discover(self, clean_data, w_v, rangex=None, sample_num=500, confidence=0.99, keepvar=False):
        assert self.attr_index < clean_data.shape[1]
        cond = None
        if rangex is not None:
            assert isinstance(rangex, Range)
            for attr in rangex.attr:
                assert attr < clean_data.shape[1]
                if cond is None:
                    cond = (clean_data[:, attr] >= rangex.l[attr]) & (clean_data[:, attr] < rangex.r[attr])
                else:
                    cond = cond & (clean_data[:, attr] >= rangex.l[attr]) & (clean_data[:, attr] < rangex.r[attr])
            timeseries = clean_data[cond, [self.attr_index]]
            self.range = rangex
        else:
            timeseries = clean_data[:, [self.attr_index]]
        var = []
        if timeseries.shape[0] - w_v > sample_num:
            sind = np.random.choice(a=timeseries.shape[0] - w_v, size=sample_num, replace=False)
        else:
            sind = np.arange(timeseries.shape[0] - w_v)
        for i in sind:
            var.append(np.var(timeseries[i: i+w_v, 0]))
        var = np.array(var)
        self.var_b = np.quantile(var, confidence)# (w-1)*np.mean(var)/st.chi2.ppf(1-confidence, w-1)
        self.hasdcv = True
        # print(self.attr_index, self.var_b)    # ***
        
        if keepvar is True:
            self.var = var

def discover_variance_constraints(clean_data, attr_indices, w_v, rangex=None,
                               sample_num=500, confidence=0.99, keepvar=False):
    variance_constraints = {}
    cond = None

    if rangex is not None:
        assert isinstance(rangex, Range)
        for attr in rangex.attr:
            assert attr < clean_data.shape[1]  # Validate attribute indices
            if cond is None:
                cond = (clean_data[:, attr] >= rangex.l[attr]) & (clean_data[:, attr] < rangex.r[attr])
            else:
                cond = cond & (clean_data[:, attr] >= rangex.l[attr]) & (clean_data[:, attr] < rangex.r[attr])
        clean_data = clean_data[cond]

    for attr_index in attr_indices:
        assert attr_index < clean_data.shape[1]  # Validate the attribute index
        vc = VarianceConstraint(attr_index)
        vc.discover(clean_data=clean_data, w_v=w_v, rangex=rangex, sample_num=sample_num, confidence=confidence, keepvar=keepvar)
        variance_constraints[attr_index] = vc.var_b

    return variance_constraints
