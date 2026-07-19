import time
import sys
from base_algorithm import BaseCleaningAlgorithm
from constraints import RowConstraintMiner, RowConstraintMiner1, ColConstraintMiner
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import minimize
import random
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List

def check_constraints_violations(data, constraints):
    violations_count = 0

    for row in data:
        for _, coefs, rho_min, rho_max in constraints:
            value = np.dot(coefs, row)  # 计算 dot product
            if not (rho_min <= value <= rho_max):  # 检查是否违反约束
                violations_count += 1
                break  # 如果这一行已经违反了某个约束，就不再检查其他约束

    return violations_count

def normalize_data(clean_data):
  
    if not isinstance(clean_data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(clean_data)
    return normalized_data, scaler

def calculate_average_absolute_difference(cleaned_data, observed_data):
    total_diff = np.abs(cleaned_data - observed_data).sum()
    n_elements = np.prod(observed_data.shape)  # 数据集中的元素总数
    average_diff = total_diff / n_elements
    return average_diff

def _adjust_scale(clean_data):
    scale_factors = {}
    for col in range(clean_data.shape[1]):
        column = clean_data[:, col]
        min_val = np.min(column)
        column -= min_val
        mean_val = np.mean(column)
        scale_factor = 10 / mean_val
        column *= scale_factor
        clean_data[:, col] = column
        scale_factors[col] = (min_val, scale_factor)
    return scale_factors

def restore_original_scale(data, scale_factors):

    restored_data = data.copy()

    for col in range(restored_data.shape[1]):
        min_val, scale_factor = scale_factors[col]
        restored_data[:, col] = (restored_data[:, col] / scale_factor) + min_val
    return restored_data

class MTSCleanRow(BaseCleaningAlgorithm):
    def __init__(self):
        
        pass

    def clean(self, clean_data, current_dirty_data, **args):
        constraints = args.get('constraints')
        if constraints is None:
            # 如果未提供 constraints，则生成默认约束
            miner = RowConstraintMiner(clean_data)
            constraints, _ = miner.mine_row_constraints(attr_num=3)

        n_rows, n_cols = current_dirty_data.shape
        cleaned_data = np.zeros_like(current_dirty_data)

        for row_idx in range(n_rows):
            row = current_dirty_data[row_idx, :]

            # 检查当前行是否违反约束条件
            violation = False
            violated_constraint = None
            for constraint_idx, (attr_indices, coefs, rho_min, rho_max) in enumerate(constraints):
                value = np.dot(coefs, row)
                if value < rho_min or value > rho_max:
                    violation = True
                    violated_constraint = (constraint_idx, attr_indices, rho_min, rho_max, value)
                    break

            if not violation:
                # 如果没有违反约束，直接保留原数据
                cleaned_data[row_idx, :] = row
                continue

            # 输出违反约束的行信息
            if violated_constraint:
                constraint_idx, attr_indices, rho_min, rho_max, value = violated_constraint
                print(f"行 {row_idx} 违反约束 {constraint_idx}:")
                print(f"  属性索引 {attr_indices}, 当前值 {value}, 范围 [{rho_min}, {rho_max}]")

            # 如果违反了约束，开始修复

            # 目标函数系数：最小化 u 和 v 的和
            c = np.hstack([np.ones(n_cols), np.ones(n_cols)])  # u 和 v 的权重都是 1

            # 构建不等式约束
            A_ub = []
            b_ub = []
            for _, coefs, rho_min, rho_max in constraints:
                # 扩展系数以适应 u 和 v
                extended_coefs = np.hstack([coefs, -coefs])

                # 添加两个不等式约束
                A_ub.append(extended_coefs)
                b_ub.append(rho_max - np.dot(coefs, row))
                A_ub.append(-extended_coefs)
                b_ub.append(-rho_min + np.dot(coefs, row))

            # 转换为 NumPy 数组
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            # 使用线性规划求解
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (n_cols * 2))

            # 处理结果
            if result.success:
                cleaned_row = result.x[:n_cols] - result.x[n_cols:] + row  # x' = u - v + x
                cleaned_data[row_idx, :] = cleaned_row
                # print(f"行 {row_idx} 违反约束后已修复: {cleaned_row}")
            else:
                print(f"线性规划求解失败于行 {row_idx}: {result.message}")
                cleaned_data[row_idx, :] = row

        # cleaned_data_df = pd.DataFrame(cleaned_data, columns=[f"col_{i}" for i in range(cleaned_data.shape[1])])
        # cleaned_data_df.to_csv("cleaned_data.csv", index=False)
        # current_dirty_data_df = pd.DataFrame(current_dirty_data, columns=[f"col_{i}" for i in range(current_dirty_data.shape[1])])
        # current_dirty_data_df.to_csv("current_dirty_data.csv", index=False)
        return cleaned_data

    @staticmethod
    def test_MTSCleanRow():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)

        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))

        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        miner = RowConstraintMiner(clean_data)
        constraints, covered_attrs = miner.mine_row_constraints(attr_num=3)

        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff = data_utils.calculate_average_absolute_difference(data_manager.clean_data, data_manager.observed_data)
        # # 输出结果
        # print("清洗前数据的平均绝对误差：", average_absolute_diff)

        # 检查数据是否违反了行约束
        violations_count = check_constraints_violations(dirty_data, constraints)
        print(f"观测数据违反的行约束次数：{violations_count}")

        # 创建MTSClean实例并清洗数据，传递constraints参数
        mtsclean = MTSCleanRow()
        cleaned_data = mtsclean.clean(clean_data, dirty_data, constraints=constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff = np.abs(cleaned_data, clean_data).sum()
        # 输出结果
        print("清洗后数据的平均绝对误差：", average_absolute_diff)

        # 检查数据是否违反了行约束
        violations_count = check_constraints_violations(cleaned_data, constraints)
        print(f"清洗数据违反的行约束次数：{violations_count}")

class MTSCleanRow_c(BaseCleaningAlgorithm):
    def __init__(self):
        # 初始化可以放置一些必要的设置
        pass

    def clean(self, clean_data, current_dirty_data, **args):
        # 从args中获取constraints，如果没有提供，则生成或处理
        constraints = args.get('constraints')

        if constraints is None:
            # 如果未提供 constraints，则生成默认约束
            miner = RowConstraintMiner1(clean_data)
            constraints, _ = miner.mine_row_constraints(attr_num=3)

        n_rows, n_cols = current_dirty_data.shape
        cleaned_data = np.zeros_like(current_dirty_data)

        for row_idx in range(n_rows):
            row = current_dirty_data[row_idx, :]

            # 检测是否违反约束
            violation = False
            violated_constraints = []

            for constraint_idx, (attr_indices, coefs, rho_min, rho_max) in enumerate(constraints):
                value = np.dot(coefs, row)
                if value < rho_min or value > rho_max:
                    violation = True
                    break  # 一旦发现违反约束，立即标记并退出检测

            if not violation:
                # 如果没有违反约束，直接保留原数据
                cleaned_data[row_idx, :] = row
                continue

            for constraint_idx, attr_indices, rho_min, rho_max, value in violated_constraints:
                print(f"行 {row_idx} 违反约束 {constraint_idx}: 属性索引 {attr_indices}, 当前值 {value}, 范围 [{rho_min}, {rho_max}]")

            # 目标函数系数：最小化 u 和 v 的和
            c = np.hstack([np.ones(n_cols), np.ones(n_cols)])  # u 和 v 的权重都是 1

            # 构建不等式约束
            A_ub = []
            b_ub = []
            for _, coefs, rho_min, rho_max in constraints:
                # 扩展系数以适应 u 和 v
                extended_coefs = np.hstack([coefs, -coefs])

                # 添加两个不等式约束
                A_ub.append(extended_coefs)
                b_ub.append(rho_max - np.dot(coefs, row))
                A_ub.append(-extended_coefs)
                b_ub.append(-rho_min + np.dot(coefs, row))

            # # 转换为 NumPy 数组
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            # 使用线性规划求解
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n_cols * 2)

            # 处理结果
            if result.success:
                cleaned_row = result.x[:n_cols] - result.x[n_cols:] + row  # x' = u - v + x
                cleaned_data[row_idx, :] = cleaned_row
            else:
                print(f"线性规划求解失败于行 {row_idx}: {result.message}")
                cleaned_data[row_idx, :] = row
        # print('clean')
        cleaned_data_df = pd.DataFrame(cleaned_data, columns=[f"col_{i}" for i in range(cleaned_data.shape[1])])
        cleaned_data_df.to_csv("cleaned_data.csv", index=False)
        current_dirty_data_df = pd.DataFrame(current_dirty_data, columns=[f"col_{i}" for i in range(current_dirty_data.shape[1])])
        current_dirty_data_df.to_csv("current_dirty_data.csv", index=False)
        return cleaned_data
    
    
    @staticmethod
    def test_MTSCleanRow_c():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)

        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))

        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        miner = RowConstraintMiner1(clean_data)
        constraints, covered_attrs = miner.mine_row_constraints(attr_num=3)

        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff = data_utils.calculate_average_absolute_difference(data_manager.clean_data, data_manager.observed_data)
        # # 输出结果
        # print("清洗前数据的平均绝对误差：", average_absolute_diff)

        # 检查数据是否违反了行约束
        violations_count = check_constraints_violations(dirty_data, constraints)
        print(f"观测数据违反的行约束次数：{violations_count}")

        # 创建MTSClean实例并清洗数据，传递constraints参数
        mtsclean = MTSCleanRow_c()
        cleaned_data = mtsclean.clean(clean_data, dirty_data, constraints=constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff = np.abs(cleaned_data, clean_data).sum()
        # 输出结果
        print("清洗后数据的平均绝对误差：", average_absolute_diff)

        # 检查数据是否违反了行约束
        violations_count = check_constraints_violations(cleaned_data, constraints)
        print(f"清洗数据违反的行约束次数：{violations_count}")


class MTSClean(BaseCleaningAlgorithm):
    def __init__(self):
        pass

    def _check_row_violations(self, row, constraints):
        """
        检查行是否违反约束。

        :param row: numpy array, 行数据
        :param constraints: list, 约束条件列表
        :return: bool, 是否违反约束
        """
        for _, coefs, rho_min, rho_max in constraints:
            value = np.dot(coefs, row)
            if value < rho_min or value > rho_max:
                return True  # 违反约束
        return False  # 未违反任何约束

    def clean(self, clean_data, current_dirty_data, **args):
       
        # 从args中获取行约束和速度约束
        row_constraints = args.get('row_constraints')
        if row_constraints is None:
            miner = RowConstraintMiner(clean_data)
            row_constraints, _ = miner.mine_row_constraints(attr_num=3)

        speed_constraints = args.get('speed_constraints')
        if speed_constraints is None:
            # 如果速度约束未提供，使用ColConstraintMiner挖掘
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints, _ = col_miner.mine_col_constraints()

        # 计算总的迭代次数
        total_steps = current_dirty_data.shape[0] * 5 + current_dirty_data.shape[0] * current_dirty_data.shape[1]
        
        with tqdm(total=total_steps, desc="MTSClean Cleaning") as pbar:
            # 先进行行约束清洗
            cleaned_data = current_dirty_data.copy()
            for _ in range(4):
                cleaned_data = self._clean_with_row_constraints(cleaned_data, row_constraints, pbar)
            # 再进行速度约束清洗
            cleaned_data = self._clean_with_speed_constraints(cleaned_data, speed_constraints, pbar)

        return cleaned_data

    def _clean_with_row_constraints(self, data, constraints, pbar):
        
        n_rows, n_cols = data.shape
        cleaned_data = np.zeros_like(data)

        for row_idx in range(n_rows):
            row = data[row_idx, :]

            # 检查是否违反约束，如果不违反则直接保留
            if not self._check_row_violations(row, constraints):
                cleaned_data[row_idx, :] = row
                pbar.update(1)
                continue

            # 目标函数系数：最小化 u 和 v 的和
            c = np.hstack([np.ones(n_cols), np.ones(n_cols)])
            A_ub = []
            b_ub = []
            for _, coefs, rho_min, rho_max in constraints:
                extended_coefs = np.hstack([coefs, -coefs])
                A_ub.append(extended_coefs)
                b_ub.append(rho_max - np.dot(coefs, row))
                A_ub.append(-extended_coefs)
                b_ub.append(-rho_min + np.dot(coefs, row))

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n_cols * 2)

            if result.success:
                cleaned_row = result.x[:n_cols] - result.x[n_cols:] + row
                cleaned_data[row_idx, :] = cleaned_row
            else:
                cleaned_data[row_idx, :] = row

            pbar.update(1)  # 更新进度条

        return cleaned_data

    def _clean_with_speed_constraints(self, data, speed_constraints, pbar):
        
        w = 10  
        cleaned_data = data.copy()
        n_rows, n_cols = data.shape

        for col in range(n_cols):
            if col not in speed_constraints:
                continue
            speed_lb, speed_ub = speed_constraints[col]
            data_col = data[:, col]

            for i in range(1, len(data_col) - 1):
                x_i_min = speed_lb + data_col[i - 1]
                x_i_max = speed_ub + data_col[i - 1]
                candidate_i = [data_col[i]]
                for k in range(i + 1, min(i + w + 1, len(data_col))):
                    candidate_i.append(speed_lb + data_col[k - 1])
                    candidate_i.append(speed_ub + data_col[k - 1])
                candidate_i = np.array(candidate_i)
                x_i_mid = np.median(candidate_i)
                if x_i_mid < x_i_min:
                    cleaned_data[i, col] = x_i_min
                elif x_i_mid > x_i_max:
                    cleaned_data[i, col] = x_i_max

                pbar.update(1)  

        return cleaned_data

    @staticmethod
    def test_MTSClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)

        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))

        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        row_miner = RowConstraintMiner(clean_data)
        row_constraints, covered_attrs = row_miner.mine_row_constraints(attr_num=3)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘速度约束
        col_miner = ColConstraintMiner(clean_data)
        speed_constraints, _ = col_miner.mine_col_constraints()

        # 检查数据是否违反了行约束
        violations_count = check_constraints_violations(dirty_data, row_constraints)
        print(f"观测数据违反的行约束次数：{violations_count}")

        # 创建MTSClean实例并清洗数据，传递行约束和速度约束参数
        mtsclean = MTSClean()
        cleaned_data = mtsclean.clean(clean_data, dirty_data, 
                                    row_constraints=row_constraints, 
                                    speed_constraints=speed_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # 检查数据是否违反了行约束
        violations_count = check_constraints_violations(cleaned_data, row_constraints)
        print(f"清洗数据违反的行约束次数：{violations_count}")

class LocalSpeedClean(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        observed_data = current_dirty_data.copy()
        speed_constraints = args.get('speed_constraints')
        w = args.get('window', 10)  

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for local speed cleaning.")

        cleaned_data = current_dirty_data.copy()

        for col in range(cleaned_data.shape[1]):
            # 获取当前列的速度约束上下界
            speed_lb = speed_constraints[col][0]
            speed_ub = speed_constraints[col][1]
            data = observed_data[:, col]

            for i in range(1, len(data) - 1):
                x_i_min = speed_lb + data[i - 1]
                x_i_max = speed_ub + data[i - 1]
                candidate_i = [data[i]]
                for k in range(i + 1, len(data)):
                    if k > i + w:
                        break
                    candidate_i.append(speed_lb + data[k])
                    candidate_i.append(speed_ub + data[k])
                candidate_i = np.array(candidate_i)
                x_i_mid = np.median(candidate_i)
                if x_i_mid < x_i_min:
                    cleaned_data[i, col] = x_i_min
                elif x_i_mid > x_i_max:
                    cleaned_data[i, col] = x_i_max

        return cleaned_data

    @staticmethod
    def test_LocalSpeedClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)
        
        # 在DataManager中注入错误
        # data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)
        # error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        # scale_factors = _adjust_scale(clean_data)
        # clean_data = restore_original_scale(clean_data, scale_factors)
        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff_before = calculate_average_absolute_difference(clean_data, dirty_data)
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        localspeedclean = LocalSpeedClean()
        cleaned_data = localspeedclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints)

        # 计算清洗后数据的平均绝对误差
        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, clean_data)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, dirty_data)
        average_absolute_diff_after = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)

class GlobalSpeedClean(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        observed_data = current_dirty_data.copy()
        speed_constraints = args.get('speed_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)  

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for global speed cleaning.")

        cleaned_data = current_dirty_data.copy()

        for col in range(observed_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col]
            x = observed_data[:, col]
            n_rows = len(x)

            for start in range(0, n_rows, chunk_length - overlap):
                end = min(start + chunk_length, n_rows)
                chunk = x[start:end]

                # 构建线性规划的目标函数和约束
                c = np.ones(len(chunk) * 2)
                A_ub = []
                b_ub = []
                for i in range(len(chunk) - 1):
                    for j in range(i + 1, len(chunk)):
                        row_diff = j - i

                        # 速度上界约束
                        A_row = np.zeros(len(chunk) * 2)
                        A_row[i] = -1
                        A_row[j] = 1
                        A_row[len(chunk) + i] = 1
                        A_row[len(chunk) + j] = -1
                        A_ub.append(A_row)
                        b_ub.append(speed_ub * row_diff - (chunk[j] - chunk[i]))

                        # 速度下界约束
                        A_row = np.zeros(len(chunk) * 2)
                        A_row[i] = 1
                        A_row[j] = -1
                        A_row[len(chunk) + i] = -1
                        A_row[len(chunk) + j] = 1
                        A_ub.append(A_row)
                        b_ub.append(-speed_lb * row_diff + (chunk[j] - chunk[i]))

                # 使用线性规划求解
                bounds = [(0, None) for _ in range(len(chunk) * 2)]
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

                # 处理结果
                if result.success:
                    cleaned_chunk = result.x[:len(chunk)] - result.x[len(chunk):] + chunk
                    cleaned_data[start:end, col] = cleaned_chunk
                else:
                    print(f"线性规划求解失败于列 {col}, 数据块 {start}-{end}: {result.message}")

        return cleaned_data

    @staticmethod
    def test_GlobalSpeedClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)
        
        # 在DataManager中注入错误
        # data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)
        # error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        # scale_factors = _adjust_scale(clean_data)
        # clean_data = restore_original_scale(clean_data, scale_factors)
        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff_before = calculate_average_absolute_difference(clean_data, dirty_data)
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        globalspeedclean = GlobalSpeedClean()
        cleaned_data = globalspeedclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints)

        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        # 计算清洗后数据的平均绝对误差
        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, clean_data)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, dirty_data)
        average_absolute_diff_after = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)

class LocalSpeedAccelClean(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        window = args.get('window', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("Both speed and acceleration constraints are required for local speed-accel cleaning.")

        cleaned_data = current_dirty_data.copy()

        # for col_idx in tqdm(range(cleaned_data.shape[1]), desc="Local Speed-Accel Cleaning"):
        for col_idx in range(cleaned_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col_idx]
            acc_lb, acc_ub = accel_constraints[col_idx]
            data = current_dirty_data[:, col_idx]

            for k in range(2, len(data) - 1):
                if not (speed_lb <= data[k] - data[k-1] <= speed_ub) and not (acc_lb <= (data[k] - data[k - 1]) - (data[k - 1] - data[k - 2]) <= acc_ub):
                    print(f"Point {k + 2} in column {col_idx} violates constraints:")
                    print(f"  Velocity: {data[k] - data[k-1]}, Acceleration: {(data[k] - data[k - 1]) - (data[k - 1] - data[k - 2])}")
                    print(f"  Speed bounds: [{speed_lb}, {speed_ub}], Accel bounds: [{acc_lb}, {acc_ub}]")
                    # break  # 找到错误点后停止运行
                    x_k_min = max(speed_lb + data[k - 1], acc_lb + data[k - 1] - data[k - 2] + data[k - 1])
                    x_k_max = min(speed_ub + data[k - 1], acc_ub + data[k - 1] - data[k - 2] + data[k - 1])

                    candidate_k = [data[k]]
                    candidate_k_min = []
                    candidate_k_max = []

                    for i in range(k + 1, len(data)):
                        if i > k + window:
                            break
                        z_k_i_a_min = (data[k - 1] * (i - k) - (acc_ub * ((i - k) ** 2) - data[i])) / (i - k + 1)
                        z_k_i_a_max = (data[k - 1] * (i - k) - (acc_lb * ((i - k) ** 2) - data[i])) / (i - k + 1)
                        z_k_i_s_min = data[i] - speed_ub * (i - k)
                        z_k_i_s_max = data[i] - speed_lb * (i - k)
                        candidate_k_min.append(min(z_k_i_s_min, z_k_i_a_min))
                        candidate_k_max.append(max(z_k_i_s_max, z_k_i_a_max))

                    candidate_k.extend(candidate_k_min)
                    candidate_k.extend(candidate_k_max)
                    candidate_k = np.array(candidate_k)
                    x_k_mid = np.median(candidate_k)

                    if x_k_mid < x_k_min:
                        cleaned_data[k, col_idx] = x_k_min
                    elif x_k_mid > x_k_max:
                        cleaned_data[k, col_idx] = x_k_max
        # print('clean')
        # cleaned_data_df = pd.DataFrame(cleaned_data, columns=[f"col_{i}" for i in range(cleaned_data.shape[1])])
        # cleaned_data_df.to_csv("cleaned_data.csv", index=False)
        # current_dirty_data_df = pd.DataFrame(current_dirty_data, columns=[f"col_{i}" for i in range(current_dirty_data.shape[1])])
        # current_dirty_data_df.to_csv("current_dirty_data.csv", index=False)
        return cleaned_data

    @staticmethod
    def test_LocalSpeedAccelClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)
        
        # 在DataManager中注入错误
        # data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)
        # error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        # scale_factors = _adjust_scale(clean_data)
        # clean_data = restore_original_scale(clean_data, scale_factors)
        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff_before = calculate_average_absolute_difference(clean_data, dirty_data)
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints, accel_constraints, _ = miner.mine_col_constraints()  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        localspeedaccelclean = LocalSpeedAccelClean()
        cleaned_data = localspeedaccelclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints, accel_constraints=accel_constraints)

        # 计算清洗后数据的平均绝对误差
        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, clean_data)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, dirty_data)
        average_absolute_diff_after = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)
    

class GlocalSpeedAccelClean(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("Both speed and acceleration constraints are required for glocal speed-accel cleaning.")

        cleaned_data = current_dirty_data.copy()
        n_rows, n_cols = current_dirty_data.shape

        # for col_idx in tqdm(range(cleaned_data.shape[1]), desc="Local Speed-Accel Cleaning"):
        for col_idx in range(cleaned_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col_idx]
            acc_lb, acc_ub = accel_constraints[col_idx]
            data = current_dirty_data[:, col_idx]

            for start in range(0, n_rows, chunk_length - overlap):
                end = min(start + chunk_length, n_rows)
                chunk = data[start:end]

                # 检测是否违反约束
                speed_violated = False
                accel_violated = False

                for i in range(len(chunk) - 1):
                    speed = abs(chunk[i + 1] - chunk[i])
                    if not (speed_lb <= speed <= speed_ub):
                        speed_violated = True

                for i in range(1, len(chunk) - 1):
                    accel = abs(chunk[i + 1] - 2 * chunk[i] + chunk[i - 1])
                    if not (acc_lb <= accel <= acc_ub):
                        accel_violated = True

                # 如果没有同时违反速度和加速度约束，继续检查
                if not (speed_violated and accel_violated):
                    continue

                print(f"违反约束: 列 {col_idx}, 数据块 {start}-{end}")
                
                c = np.ones(2 * len(chunk))
                A_ub = []
                b_ub = []
                bounds = [(0, None) for _ in range(2 * len(chunk))]

                for i in range(len(chunk)):
                    for j in range(i + 1, len(chunk)):

                        bij_max = speed_ub * (j - i) - (chunk[j] - chunk[i])
                        bij_min = -speed_lb * (j - i) + (chunk[j] - chunk[i])

                        # 速度约束
                        A_row_max = np.zeros(2 * len(chunk))
                        A_row_min = np.zeros(2 * len(chunk))
                        A_row_max[j], A_row_max[i] = 1, -1
                        A_row_max[j + len(chunk)], A_row_max[i + len(chunk)] = -1, 1
                        A_row_min[j], A_row_min[i] = -1, 1
                        A_row_min[j + len(chunk)], A_row_min[i + len(chunk)] = 1, -1
                        A_ub.append(A_row_max)
                        b_ub.append(bij_max)
                        A_ub.append(A_row_min)
                        b_ub.append(bij_min)

                        # 加速度约束
                        if i >= 1:
                            bij_max = acc_ub * (j - i) - (chunk[j] - chunk[i]) / (j - i) + (chunk[i] - chunk[i - 1])
                            bij_min = -acc_lb * (j - i) + (chunk[j] - chunk[i]) / (j - i) - (chunk[i] - chunk[i - 1])
                            A_row_max = np.zeros(2 * len(chunk))
                            A_row_min = np.zeros(2 * len(chunk))
                            tmp1, tmp2 = 1 / (j - i), 1
                            A_row_max[j], A_row_max[j + len(chunk)] = tmp1, -tmp1
                            A_row_max[i], A_row_max[i + len(chunk)] = -tmp1 - tmp2, tmp1 + tmp2
                            A_row_max[i - 1], A_row_max[i - 1 + len(chunk)] = tmp2, -tmp2
                            A_row_min[j], A_row_min[j + len(chunk)] = -tmp1, tmp1
                            A_row_min[i], A_row_min[i + len(chunk)] = tmp1 + tmp2, -tmp1 - tmp2
                            A_row_min[i - 1], A_row_min[i - 1 + len(chunk)] = -tmp2, tmp2
                            A_ub.append(A_row_max)
                            b_ub.append(bij_max)
                            A_ub.append(A_row_min)
                            b_ub.append(bij_min)

                # 求解线性规划
                result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds)
                if result.success:
                    cleaned_chunk = result.x[:len(chunk)] - result.x[len(chunk):] + chunk
                    cleaned_data[start:end, col_idx] = cleaned_chunk
                else:
                    print(f"线性规划求解失败于列 {col_idx}, 数据块 {start}-{end}: {result.message}")
        # print('zai')
        # cleaned_data_df = pd.DataFrame(cleaned_data, columns=[f"col_{i}" for i in range(cleaned_data.shape[1])])
        # cleaned_data_df.to_csv("cleaned_data.csv", index=False)
        # current_dirty_data_df = pd.DataFrame(current_dirty_data, columns=[f"col_{i}" for i in range(current_dirty_data.shape[1])])
        # current_dirty_data_df.to_csv("current_dirty_data.csv", index=False)
        return cleaned_data

    @staticmethod
    def test_GlocalSpeedAccelClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)
        
        # 在DataManager中注入错误
        # data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)
        # error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        # scale_factors = _adjust_scale(clean_data)
        # clean_data = restore_original_scale(clean_data, scale_factors)
        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff_before = calculate_average_absolute_difference(clean_data, dirty_data)
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints, accel_constraints = miner.mine_col_constraints()  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        glocalspeedaccelclean = GlocalSpeedAccelClean()
        cleaned_data = glocalspeedaccelclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints, accel_constraints=accel_constraints)

        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        # 计算清洗后数据的平均绝对误差
        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, clean_data)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, dirty_data)
        average_absolute_diff_after = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)

class VarianceConstraintClean(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        """
        Clean the dataset based on variance constraints.

        Parameters:
            clean_data (np.ndarray): The array to store cleaned data.
            current_dirty_data (np.ndarray): The input dirty data array.
            **args: Additional arguments including variance constraints and parameters.
        """
        variance_constraints = args.get('variance_constraints')
        window = args.get('window', 10)
        beta = args.get('beta', 0.5)
        # attr_indices = args.get('attr_indices', [])

        if variance_constraints is None:
            raise ValueError("Variance constraints are required for cleaning.")

        cleaned_data = current_dirty_data.copy()

        for col_idx in range(cleaned_data.shape[1]):
            if col_idx not in variance_constraints:
                continue  # Skip columns that don't have variance constraints

            sequence = cleaned_data[:, col_idx]
            n = sequence.shape[0]

            # Clean sequence with variance constraint
            for k in range(window - 1, n - 2 * window):
                repair_sum = 0
                weight_sum = 0

                for i in range(k - window + 1, k + 1):
                    num = i - (k - window + 1)
                    weight_sum += pow(beta, num)

                    # Check if variance within window is within the constraint
                    if np.var(sequence[i:i + window]) <= variance_constraints[col_idx]:
                        repair_sum += pow(beta, num) * sequence[k]
                    else:
                        # Solve quadratic equation to find possible values
                        l1_sum = np.sum(sequence[i:i + window]) - sequence[k]
                        l2_sum = np.sum(sequence[i:i + window] ** 2) - sequence[k] ** 2
                        x1, x2 = self.solve_quad(
                            window - 1, -2 * l1_sum, window * l2_sum - l1_sum ** 2 - window * window * variance_constraints[col_idx]
                        )

                        if x1 is None:
                            continue
                        elif abs(x1 - sequence[k]) > abs(x2 - sequence[k]):
                            repair_sum += pow(beta, num) * x2
                        else:
                            repair_sum += pow(beta, num) * x1

                # Compute the final repaired value
                repair = repair_sum / weight_sum
                if sequence[k] != repair:
                    sequence[k] = repair

            # Update the cleaned data with the repaired sequence
            cleaned_data[:, col_idx] = sequence
        print('comp')
        return cleaned_data

    def solve_quad(self, A, B, C):
        delta = B ** 2 - 4 * A * C  # 计算delta
        if delta <= 0:  # 判断delta如果小于0
            x = B / (-2 * A)  # 则解为B / (-2*A)
            return x, x
        else:  # 除以上两种情况以外，即delta如果大于0
            x1 = (B + delta ** 0.5) / (-2 * A)  # 计算解1：x1
            x2 = (B - delta ** 0.5) / (-2 * A)  # 计算解2：x2
            return x1, x2

    def test_VarianceConstraintClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)
        
        # 在DataManager中注入错误
        # data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)
        # error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        print('dirty', dirty_data.shape)
        print('clean', clean_data.shape)
        
        # scale_factors = _adjust_scale(clean_data)
        # clean_data = restore_original_scale(clean_data, scale_factors)
        # 计算清洗前数据的平均绝对误差
        # average_absolute_diff_before = calculate_average_absolute_difference(clean_data, dirty_data)
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        _, _, variance_constraints = miner.mine_col_constraints()  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        varianceclean = VarianceConstraintClean()
        cleaned_data = varianceclean.clean(clean_data, dirty_data, variance_constraints=variance_constraints)
        print('cleaned', cleaned_data.shape)

        # 计算清洗后数据的平均绝对误差
        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, clean_data)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, dirty_data)
        average_absolute_diff_after = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)

def solve_quad(A, B, C):
    delta = B ** 2 - 4 * A * C  # 计算delta
    if delta <= 0:  # 判断delta如果小于0
        x = B / (-2 * A)  # 则解为B / (-2*A)
        return x, x
    else:  # 除以上两种情况以外，即delta如果大于0
        x1 = (B + delta ** 0.5) / (-2 * A)  # 计算解1：x1
        x2 = (B - delta ** 0.5) / (-2 * A)  # 计算解2：x2
        return x1, x2

class DataCleaner:

    @staticmethod
    def variance_constraint_clean(clean_data, dirty_data, variance_constraints, attr_indices, w=10, beta=0.5):
        """
        Applies variance constraints on the given clean data for specific attributes.
        """
        cleaned_data = np.copy(dirty_data)
        print(cleaned_data)

        for attr_index in attr_indices:
            assert attr_index < cleaned_data.shape[1]
            assert attr_index in variance_constraints.keys()

            # Assume the first w points don't violate variance constraints
            for k in range(w - 1, dirty_data.shape[0] - 2 * w):
                repair_sum = 0
                weight_sum = 0
                for i in range(k - w + 1, k + 1):
                    num = i - (k - w + 1)
                    weight_sum += pow(beta, num)

                    if np.var(cleaned_data[i:i + w, attr_index]) <= variance_constraints[attr_index]:
                        repair_sum += pow(beta, num) * cleaned_data[k, attr_index]
                    else:
                        # Calculate L1 and L2 sums for quadratic equation
                        l1_sum = np.sum(cleaned_data[i:i + w, attr_index]) - cleaned_data[k, attr_index]
                        l2_sum = (
                            np.sum(cleaned_data[i:i + w, attr_index] ** 2)
                            - cleaned_data[k, attr_index] ** 2
                        )
                        x1, x2 = solve_quad(
                            w - 1,
                            -2 * l1_sum,
                            w * l2_sum - l1_sum * l1_sum - w * w * variance_constraints[attr_index],
                        )
                        if x1 is None:
                            continue
                        else:
                            # Choose the root that differs most from the current value
                            if abs(x1 - cleaned_data[k, attr_index]) > abs(x2 - cleaned_data[k, attr_index]):
                                repair_sum += pow(beta, num) * x2
                            else:
                                repair_sum += pow(beta, num) * x1

                # Calculate repair value
                repair = repair_sum / weight_sum if weight_sum != 0 else cleaned_data[k, attr_index]
                if repair != cleaned_data[k, attr_index]:
                    print(f"Repairing index {k}: Old value = {cleaned_data[k, attr_index]}, New value = {repair}")
                    cleaned_data[k, attr_index] = repair

        print('Cleaning complete.')
        return cleaned_data

    @staticmethod
    def test_variance_constraint_clean():
        """
        Tests the `variance_constraint_clean` method.
        """
        # Load data
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        # clean_data = np.array(clean_data_df.values)
        # clean_data = clean_data.astype(np.float64)
        clean_data = clean_data_df.values.astype(np.float64)
        
        # Inject errors into the clean data
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        # Calculate the average absolute difference before cleaning
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        

        # Use ColConstraintMiner to mine column constraints
        # attr_indices = range(clean_data.shape[1])
        attr_indices = range(clean_data.shape[1])
        variance_constraints = discover_variance_constraints(clean_data, attr_indices, w=10)

        cleaned_data = DataCleaner.variance_constraint_clean(clean_data, dirty_data, variance_constraints, attr_indices)

        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        # Calculate the average absolute difference after cleaning
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # Calculate the average repair distance (difference between dirty and cleaned data)
        average_absolute_diff_after_repair = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after_repair)

class DataBase:
    def __init__(self, dataframe):
        self.dataframe = dataframe

class VariDataCleaner1:
    @staticmethod
    def variance_constraint_clean(db: DataBase, variance_constraints: dict, ra: list, w_v=10, beta=0.5):
        # assert t_attr in db.dataframe.columns
        for attr in ra:
            assert attr in db.dataframe.columns
            assert attr in variance_constraints.keys()
            sequence = np.array(db.dataframe[attr])

            for k in range(w_v - 1, db.dataframe.shape[0] - 2 * w_v):
                repair_sum = 0
                weight_sum = 0
                for i in range(k - w_v + 1, k + 1):
                    num = i - (k - w_v + 1)
                    weight_sum += pow(beta, num)
                    if np.var(sequence[i: i + w_v]) <= variance_constraints[attr]:
                        repair_sum += pow(beta, num) * sequence[k]
                        # print(beta)
                        # print(num)
                        # print(sequence[k])
                        # print(repair_sum)
                        # coding
                    else:
                        l1_sum = np.sum(sequence[i: i + w_v]) - sequence[k]
                        l2_sum = np.sum(sequence[i: i + w_v] * sequence[i: i + w_v]) - sequence[k] * sequence[k]
                        x1, x2 = solve_quad(w_v - 1, -2 * l1_sum, w_v * l2_sum - l1_sum * l1_sum - w_v * w_v * variance_constraints[attr])
                        if x1 is None:
                            continue
                        elif abs(x1 - sequence[k]) > abs(x2 - sequence[k]):
                            repair_sum += pow(beta, num) * x2
                        else:
                            repair_sum += pow(beta, num) * x1
                repair = repair_sum / weight_sum
                if sequence[k] != repair:
                    sequence[k] = repair
            db.dataframe[attr] = sequence
        return db.dataframe.values

    @staticmethod
    def test_variance_constraint_clean():
        # Load data
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = clean_data_df.values.astype(np.float64)

        # Inject errors into the clean data
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))

        # Calculate the average absolute difference before cleaning
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()

        # Use ColConstraintMiner to mine column constraints
        attr_indices = range(clean_data.shape[1])
        variance_constraints = discover_variance_constraints(clean_data, attr_indices, w=10)

        # Perform cleaning
        db = DataBase(pd.DataFrame(dirty_data))
        cleaned_data = DataCleaner1.variance_constraint_clean(db, variance_constraints, attr_indices, w=10, beta=0.5)
        # cleaned_data = db.dataframe.values
        print(cleaned_data.shape)
        print(clean_data.shape)
        print(dirty_data.shape)

        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        # Calculate the average absolute difference after cleaning
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # Calculate the average repair distance (difference between dirty and cleaned data)
        average_absolute_diff_after_repair = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after_repair)

