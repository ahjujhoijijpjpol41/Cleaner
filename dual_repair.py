import time
import sys
from base_algorithm import BaseCleaningAlgorithm
from dual_constraints import RowConstraintMiner, RowConstraintMiner1, ColConstraintMiner, ColConstraintMiner, discover_variance_constraints
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import minimize
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
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

class MTSCleanRow_repair(BaseCleaningAlgorithm):
    def __init__(self):
   
        pass

    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        constraints = args.get('constraints')
        if constraints is None:
            # 如果未提供 constraints，则生成默认约束
            miner = RowConstraintMiner(clean_data)
            constraints, _ = miner.mine_row_constraints(attr_num=3)

        n_rows, n_cols = current_dirty_data.shape
        cleaned_data = np.copy(current_dirty_data)  # 初始数据直接用原始数据

        for row_idx in range(n_rows):
            if detect_errors[row_idx].sum() == 0:
                # 如果当前行没有错误（detect_errors 中全为 0），直接跳过
                continue
            
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
                # 如果修复失败，保留原始行数据
                cleaned_data[row_idx, :] = row

        # 对于 detect_errors 中为 0 的行，不进行任何修改，直接保留原数据
        for row_idx in range(n_rows):
            if detect_errors[row_idx].sum() == 0:
                continue  # 跳过没有错误的行

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


class LocalSpeedClean_repair(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        observed_data = current_dirty_data.copy()
        speed_constraints = args.get('speed_constraints')
        w = args.get('window', 10)  

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for local speed cleaning.")

        # 创建清洗后的数据副本
        cleaned_data = current_dirty_data.copy()

        for col in range(cleaned_data.shape[1]):
            # 获取当前列的速度约束上下界
            speed_lb = speed_constraints[col][0]
            speed_ub = speed_constraints[col][1]
            data = observed_data[:, col]

            for i in range(1, len(data) - 1):
                # Only apply cleaning where detect_errors is 1
                if detect_errors[i, col] == 1:
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
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        # 使用假设的 error detection 来模拟错误位置
        detect_errors = np.random.choice([0, 1], size=dirty_data.shape, p=[0.9, 0.1])

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  # 获取速度约束

        # 创建LocalSpeedClean实例并清洗数据，传递速度约束参数
        localspeedclean = LocalSpeedClean()
        cleaned_data = localspeedclean.clean(clean_data, dirty_data, detect_errors=detect_errors, speed_constraints=speed_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        average_absolute_diff_after_dirty = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after_dirty)

class GlobalSpeedClean_repair(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        
        observed_data = current_dirty_data.copy()
        speed_constraints = args.get('speed_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)  

        if speed_constraints is None:
            raise ValueError("Speed constraints被global speed cleaning需要")

        
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
                else:
                    print(f"线性规划求解失败于列 {col}, 数据块 {start}-{end}: {result.message}")
                    cleaned_chunk = chunk  # 如果求解失败，保持原始数据

                # 如果 detect_errors 为 1，则修复该位置的脏数据
                for i in range(len(chunk)):
                    if detect_errors[start + i, col] == 1:
                        cleaned_data[start + i, col] = cleaned_chunk[i]
                    # 如果 detect_errors[start + i, col] == 0, 保持 cleaned_data 中的原始数据

        return cleaned_data

    @staticmethod
    def test_GlobalSpeedClean():
        # 加载数据
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)

        # 在DataManager中注入错误
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  # 获取速度约束

        # 创建GlobalSpeedClean实例并清洗数据，传递速度约束参数
        globalspeedclean = GlobalSpeedClean()
        detect_errors = np.random.randint(0, 2, size=dirty_data.shape)  # 模拟一个错误检测标记矩阵
        cleaned_data = globalspeedclean.clean(clean_data, dirty_data, detect_errors=detect_errors, speed_constraints=speed_constraints)

        # 输出修复后的数据
        print("修复后的数据：")
        print(cleaned_data)

        # 计算修复前数据和修复后数据之间的误差
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

class LocalSpeedAccelClean_repair(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        window = args.get('window', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("speed and acceleration constraints被local speed-accel cleaning需要")

        cleaned_data = current_dirty_data.copy()

        for col_idx in range(current_dirty_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col_idx]
            acc_lb, acc_ub = accel_constraints[col_idx]
            data = current_dirty_data[:, col_idx]

            for k in range(2, len(data) - 1):
                if detect_errors[k, col_idx] == 1:  # Only process points marked as errors
                    if not (speed_lb <= data[k] - data[k-1] <= speed_ub) and not (acc_lb <= (data[k] - data[k - 1]) - (data[k - 1] - data[k - 2]) <= acc_ub):
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
                        else:
                            cleaned_data[k, col_idx] = x_k_mid

                else:
                    cleaned_data[k, col_idx] = current_dirty_data[k, col_idx]  # Retain original clean data for valid points

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
        speed_constraints, accel_constraints = miner.mine_col_constraints()  # 获取速度和加速度约束

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

class GlocalSpeedAccelClean_repair(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("speed and acceleration constraints被glocal speed-accel cleaning需要")

        # 初始化修复后的数据
        cleaned_data = current_dirty_data.copy()  # 使用clean_data的原始数据为基础
        n_rows, n_cols = current_dirty_data.shape

        # 遍历每一列数据
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
                else:
                    print(f"线性规划求解失败于列 {col_idx}, 数据块 {start}-{end}: {result.message}")
                    cleaned_chunk = chunk  # 如果求解失败，保持原始数据

                # 将修复后的数据替换到对应的错误位置
                for i in range(len(chunk)):
                    if detect_errors[start + i, col_idx] == 1:
                        cleaned_data[start + i, col_idx] = cleaned_chunk[i]
                    # 如果 detect_errors[start + i, col_idx] == 0, 保持clean_data中的原值

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
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(clean_data)
        speed_constraints, accel_constraints = miner.mine_col_constraints()  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        glocalspeedaccelclean = GlocalSpeedAccelClean()
        cleaned_data = glocalspeedaccelclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints, accel_constraints=accel_constraints)

        # 计算清洗后数据的平均绝对误差
        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, clean_data)
        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # average_absolute_diff_after = calculate_average_absolute_difference(cleaned_data, dirty_data)
        average_absolute_diff_after = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)

class VarianceConstraintClean_repair(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        
        variance_constraints = args.get('variance_constraints')
        window = args.get('window', 10)
        beta = args.get('beta', 0.5)

        if variance_constraints is None:
            raise ValueError("Variance constraints 被cleaning需要")

        cleaned_data = current_dirty_data.copy()

        for col_idx in range(cleaned_data.shape[1]):
            if col_idx not in variance_constraints:
                continue  

            sequence = cleaned_data[:, col_idx]
            error_mask = detect_errors[:, col_idx] 
            n = sequence.shape[0]

            for k in range(window - 1, n - 2 * window):
                if error_mask[k] == 0:
                    continue  

                repair_sum = 0
                weight_sum = 0

                for i in range(k - window + 1, k + 1):
                    num = i - (k - window + 1)
                    weight_sum += pow(beta, num)

                    if np.var(sequence[i:i + window]) <= variance_constraints[col_idx]:
                        repair_sum += pow(beta, num) * sequence[k]
                    else:
                        l1_sum = np.sum(sequence[i:i + window]) - sequence[k]
                        l2_sum = np.sum(sequence[i:i + window] ** 2) - sequence[k] ** 2
                        x1, x2 = self.solve_quad(
                            window - 1, -2 * l1_sum,
                            window * l2_sum - l1_sum ** 2 - window * window * variance_constraints[col_idx]
                        )

                        if x1 is None:
                            continue
                        elif abs(x1 - sequence[k]) > abs(x2 - sequence[k]):
                            repair_sum += pow(beta, num) * x2
                        else:
                            repair_sum += pow(beta, num) * x1

                repair = repair_sum / weight_sum
                sequence[k] = repair  

           
            cleaned_data[:, col_idx] = sequence

        return cleaned_data

    def solve_quad(self, A, B, C):
        delta = B ** 2 - 4 * A * C
        if delta <= 0:
            x = B / (-2 * A)
            return x, x
        else:
            x1 = (B + delta ** 0.5) / (-2 * A)
            x2 = (B - delta ** 0.5) / (-2 * A)
            return x1, x2

    @staticmethod
    def test_VarianceConstraintClean():
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values).astype(np.float64)
        
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        detect_errors = detect_errors_method(dirty_data, clean_data)  

        miner = ColConstraintMiner(clean_data)
        _, _, variance_constraints = miner.mine_col_constraints()

        variance_cleaner = VarianceConstraintClean()
        cleaned_data = variance_cleaner.clean(clean_data, dirty_data, detect_errors, variance_constraints=variance_constraints)

        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

class DataBase:
    def __init__(self, dataframe):
        self.dataframe = dataframe


def solve_quad(A, B, C):
    delta = B ** 2 - 4 * A * C  # 计算delta
    if delta <= 0:  # 判断delta如果小于0
        x = B / (-2 * A)  # 则解为B / (-2*A)
        return x, x
    else:  # 除以上两种情况以外，即delta如果大于0
        x1 = (B + delta ** 0.5) / (-2 * A)  # 计算解1：x1
        x2 = (B - delta ** 0.5) / (-2 * A)  # 计算解2：x2
        return x1, x2

class VariDataCleaner1_repair:
    @staticmethod
    def variance_constraint_clean(db: DataBase, variance_constraints: dict, ra: list, detect_errors: np.ndarray, w_v=10, beta=0.5):
        # 遍历属性列表 (ra)
        for attr in ra:
            assert attr in db.dataframe.columns
            assert attr in variance_constraints.keys()

            # 获取当前属性的索引
            attr_idx = db.dataframe.columns.get_loc(attr)
            sequence = np.array(db.dataframe[attr])

            # 按照窗口大小 w_v 遍历数据
            for k in range(w_v - 1, db.dataframe.shape[0] - 2 * w_v):
                # 提取与当前属性相关的检测结果
                error_flag = detect_errors[k, attr_idx]  
                if error_flag == 1:  # 仅当检测到错误时才进行修复
                    repair_sum = 0
                    weight_sum = 0
                    for i in range(k - w_v + 1, k + 1):
                        num = i - (k - w_v + 1)
                        weight_sum += pow(beta, num)

                        # 检查方差约束是否满足
                        if np.var(sequence[i: i + w_v]) <= variance_constraints[attr]:
                            repair_sum += pow(beta, num) * sequence[k]
                        else:
                            l1_sum = np.sum(sequence[i: i + w_v]) - sequence[k]
                            l2_sum = np.sum(sequence[i: i + w_v] * sequence[i: i + w_v]) - sequence[k] * sequence[k]
                            # 解二次方程
                            x1, x2 = solve_quad(w_v - 1, -2 * l1_sum, w_v * l2_sum - l1_sum * l1_sum - w_v * w_v * variance_constraints[attr])
                            if x1 is None:
                                continue
                            elif abs(x1 - sequence[k]) > abs(x2 - sequence[k]):
                                repair_sum += pow(beta, num) * x2
                            else:
                                repair_sum += pow(beta, num) * x1

                    # 计算修复值并更新到序列中
                    repair = repair_sum / weight_sum
                    sequence[k] = repair
            # 更新当前属性的修复后的数据到数据框中
            db.dataframe[attr] = sequence

        return db.dataframe.values

    @staticmethod
    def test_variance_constraint_clean():
        
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = clean_data_df.values.astype(np.float64)

        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))

        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()

        attr_indices = range(clean_data.shape[1])
        variance_constraints = discover_variance_constraints(clean_data, attr_indices, w=10)


        detect_errors = np.zeros(dirty_data.shape[0], dtype=int)

        detect_errors[10:20] = 1  
        
        db = DataBase(pd.DataFrame(dirty_data))
        cleaned_data = VariDataCleaner1.variance_constraint_clean(db, variance_constraints, attr_indices, detect_errors, w=10, beta=0.5)

        print(cleaned_data.shape)
        print(clean_data.shape)
        print(dirty_data.shape)

        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        average_absolute_diff_after = np.abs(cleaned_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        average_absolute_diff_after_repair = np.abs(cleaned_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after_repair)


class LinearInterpolationRepair(BaseCleaningAlgorithm):

    def __init__(self):
        pass
    
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
   
        n_rows, n_cols = current_dirty_data.shape
        repaired_data = current_dirty_data.copy()
        
        for col in range(n_cols):
            col_data = repaired_data[:, col]
            error_indices = np.where(detect_errors[:, col] == 1)[0]
            
            if len(error_indices) == 0:
                continue
            
            # 获取非错误点的索引和值
            good_indices = np.where(detect_errors[:, col] == 0)[0]
            
            if len(good_indices) == 0:
               
                col_data[:] = np.mean(col_data)
                continue
            
            # 对每个错误点进行线性插值
            for idx in error_indices:
                # 找到前后最近的非错误点
                prev_good = good_indices[good_indices < idx]
                next_good = good_indices[good_indices > idx]
                
                if len(prev_good) == 0:
                
                    if len(next_good) > 0:
                        col_data[idx] = col_data[next_good[0]]
                    else:
                        col_data[idx] = np.mean(col_data[good_indices])
                elif len(next_good) == 0:
           
                    col_data[idx] = col_data[prev_good[-1]]
                else:
                    # 线性插值
                    x0, x1 = prev_good[-1], next_good[0]
                    y0, y1 = col_data[x0], col_data[x1]
                    col_data[idx] = y0 + (y1 - y0) * (idx - x0) / (x1 - x0)
            
            repaired_data[:, col] = col_data
        
        return repaired_data


class KNNRepair(BaseCleaningAlgorithm):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):
        n_rows, n_cols = current_dirty_data.shape
        repaired_data = current_dirty_data.copy()
        
        # 对每一列分别修复
        for col in range(n_cols):
            col_data = repaired_data[:, col]
            error_indices = np.where(detect_errors[:, col] == 1)[0]
            
            if len(error_indices) == 0:
                continue
            
            good_indices = np.where(detect_errors[:, col] == 0)[0]
            
            if len(good_indices) == 0:
                repaired_data[:, col] = np.mean(current_dirty_data[:, col])
                continue
            
            X_train = current_dirty_data[good_indices, :]
            X_train = np.delete(X_train, col, axis=1)
            
            if X_train.shape[1] == 0:
                X_train = good_indices.reshape(-1, 1)
                X_test = error_indices.reshape(-1, 1)
            else:
                X_test = current_dirty_data[error_indices, :]
                X_test = np.delete(X_test, col, axis=1)
            
            y_train = col_data[good_indices]
            
            k = min(self.n_neighbors, len(good_indices))
            if k == 0:
                continue
            
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train, y_train)
            
            if len(error_indices) > 0:
                predicted_values = knn.predict(X_test)
                repaired_data[error_indices, col] = predicted_values
        
        return repaired_data


class MultiColumnKNNRepair(BaseCleaningAlgorithm):

    def __init__(self, n_neighbors=5):
        """
        参数:
            n_neighbors: K值
        """
        self.n_neighbors = n_neighbors
    
    def clean(self, clean_data, current_dirty_data, detect_errors, **args):

        n_rows, n_cols = current_dirty_data.shape
        repaired_data = current_dirty_data.copy()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(current_dirty_data)

        has_error = np.any(detect_errors == 1, axis=1)
        good_indices = np.where(~has_error)[0]
        error_indices = np.where(has_error)[0]
        
        if len(good_indices) == 0:

            for col in range(n_cols):
                repaired_data[:, col] = np.mean(current_dirty_data[:, col])
            return repaired_data
        
        if len(error_indices) == 0:
            return repaired_data
        
        X_train = data_scaled[good_indices, :]
        y_train = X_train

        X_test = data_scaled[error_indices, :]

        k = min(self.n_neighbors, len(good_indices))
        
        if k == 0:
            return repaired_data
        
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # 预测错误行的值
        if len(error_indices) > 0:
            predicted_values = knn.predict(X_test)
            repaired_data[error_indices, :] = scaler.inverse_transform(predicted_values)
        
        return repaired_data
