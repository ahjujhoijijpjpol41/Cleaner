import time
import sys
from base_algorithm import BaseCleaningAlgorithm
from dual_constraints import RowConstraintMiner, RowConstraintMiner1, ColConstraintMiner, discover_variance_constraints
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import minimize
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
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

class MTSCleanRow_detection(BaseCleaningAlgorithm):
    def __init__(self):
        
        pass

    def clean(self, clean_data, current_dirty_data, **args):
        constraints = args.get('constraints')
        if constraints is None:
            # 如果未提供 constraints，则生成默认约束
            miner = RowConstraintMiner(clean_data)
            constraints, _ = miner.mine_row_constraints(attr_num=3)

        n_rows, n_cols = current_dirty_data.shape
        error_mask = np.zeros_like(current_dirty_data, dtype=int)  # 创建一个 error_mask 来标记是否违反约束

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

            if violation:
                # 如果违反了约束，标记错误位置
                if violated_constraint:
                    constraint_idx, attr_indices, rho_min, rho_max, value = violated_constraint
                    print(f"行 {row_idx} 违反约束 {constraint_idx}:")
                    print(f"  属性索引 {attr_indices}, 当前值 {value}, 范围 [{rho_min}, {rho_max}]")
                
                # 设置 error_mask 为 1，表示这一行违反了约束
                error_mask[row_idx, :] = 1
            # else:
            #     # 如果没有违反约束，标记没有错误的行
            #     error_mask[row_idx, :] = 0

        return error_mask

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


class LocalSpeedClean_detection(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        observed_data = current_dirty_data.copy()
        speed_constraints = args.get('speed_constraints')
        w = args.get('window', 10) 

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for local speed cleaning.")

        error_mask = np.zeros(current_dirty_data.shape, dtype=int)

        for col in range(observed_data.shape[1]):
          
            speed_lb = speed_constraints[col][0]
            speed_ub = speed_constraints[col][1]
            data = observed_data[:, col]

           
            for i in range(1, len(data) - 1): 
                
                if data[i] < speed_lb or data[i] > speed_ub:
                    error_mask[i, col] = 1  

        
        return error_mask

    @staticmethod
    def test_LocalSpeedClean():
      
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values).astype(np.float64)
        
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        
        
        miner = ColConstraintMiner(clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  

        localspeedclean = LocalSpeedClean()
        error_mask = localspeedclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints)
       
        print("Error Mask:")
        print(error_mask)

        print(f"Number of errors detected: {np.sum(error_mask)}")

class GlobalSpeedClean_detection(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        
        observed_data = current_dirty_data.copy()
        speed_constraints = args.get('speed_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)  

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for global speed cleaning.")

        # 初始化 error_mask，大小与 current_dirty_data 相同，初始值为0
        error_mask = np.zeros(current_dirty_data.shape, dtype=int)

        for col in range(observed_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col]
            x = observed_data[:, col]
            n_rows = len(x)

            for start in range(0, n_rows, chunk_length - overlap):
                end = min(start + chunk_length, n_rows)
                chunk = x[start:end]

                # 检查速度约束 violations
                for i in range(len(chunk) - 1):
                    speed = abs(chunk[i + 1] - chunk[i])
                    if speed < speed_lb or speed > speed_ub:
                        # 如果违反速度约束，更新 error_mask
                        for j in range(i, len(chunk)):
                            error_mask[start + j, col] = 1

        return error_mask

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

        # 创建GlobalSpeedClean实例并检查速度约束的错误
        globalspeedclean = GlobalSpeedClean()
        error_mask = globalspeedclean.clean(clean_data, dirty_data, speed_constraints=speed_constraints)

        # 输出错误掩码
        print("Error Mask (Violations of speed constraints):")
        print(error_mask)

        # 计算原始数据和脏数据之间的误差
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

class LocalSpeedAccelClean_detection(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        window = args.get('window', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("Both speed and acceleration constraints are required for local speed-accel cleaning.")

        error_mask = np.zeros(current_dirty_data.shape, dtype=int)

        for col_idx in range(current_dirty_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col_idx]
            acc_lb, acc_ub = accel_constraints[col_idx]
            data = current_dirty_data[:, col_idx]

            for k in range(2, len(data) - 1):
              
                if not (speed_lb <= data[k] - data[k-1] <= speed_ub) and not (acc_lb <= (data[k] - data[k - 1]) - (data[k - 1] - data[k - 2]) <= acc_ub):
                    print(f"Point {k + 2} in column {col_idx} violates constraints:")
                    print(f"  Velocity: {data[k] - data[k-1]}, Acceleration: {(data[k] - data[k - 1]) - (data[k - 1] - data[k - 2])}")
                    print(f"  Speed bounds: [{speed_lb}, {speed_ub}], Accel bounds: [{acc_lb}, {acc_ub}]")

                    error_mask[k, col_idx] = 1

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

        
        return error_mask


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


class GlocalSpeedAccelClean_detection(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("speed and acceleration constraints都被glocal speed-accel cleaning需要")

        n_rows, n_cols = current_dirty_data.shape
        error_mask = np.zeros(current_dirty_data.shape, dtype=int)

        for col_idx in range(current_dirty_data.shape[1]):
            speed_lb, speed_ub = speed_constraints[col_idx]
            acc_lb, acc_ub = accel_constraints[col_idx]
            data = current_dirty_data[:, col_idx]

            for start in range(0, n_rows, chunk_length - overlap):
                end = min(start + chunk_length, n_rows)
                chunk = data[start:end]

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

                if speed_violated and accel_violated:
                    print(f"Violation detected: Column {col_idx}, Data Chunk {start}-{end}")

                    for i in range(start, end):
                        error_mask[i, col_idx] = 1

        return error_mask

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

class VarianceConstraintClean_detection(BaseCleaningAlgorithm):
    def clean(self, clean_data, current_dirty_data, **args):
      
        variance_constraints = args.get('variance_constraints')
        window = args.get('window', 10)
        beta = args.get('beta', 0.5)

        if variance_constraints is None:
            raise ValueError("Variance constraints are required for cleaning.")

        error_mask = np.zeros(current_dirty_data.shape[0], dtype=int)

        for col_idx in range(current_dirty_data.shape[1]):
            if col_idx not in variance_constraints:
                continue  

            sequence = current_dirty_data[:, col_idx]
            n = sequence.shape[0]

            for k in range(window - 1, n - 2 * window):

                if np.var(sequence[k - window + 1:k + 1]) > variance_constraints[col_idx]:
                    error_mask[k] = 1  
        print('comp')
        return error_mask

    def solve_quad(self, A, B, C):
        delta = B ** 2 - 4 * A * C 
        if delta <= 0:  
            x = B / (-2 * A)  
            return x, x
        else:
            x1 = (B + delta ** 0.5) / (-2 * A)  
            x2 = (B - delta ** 0.5) / (-2 * A)  
            return x1, x2

    def test_VarianceConstraintClean():
        
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = np.array(clean_data_df.values)
        clean_data = clean_data.astype(np.float64)
        
       
        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))
        print('dirty', dirty_data.shape)
        print('clean', clean_data.shape)
        
        
        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        miner = ColConstraintMiner(clean_data)
        _, _, variance_constraints = miner.mine_col_constraints()  

        varianceclean = VarianceConstraintClean()
        error_mask = varianceclean.clean(clean_data, dirty_data, variance_constraints=variance_constraints)
        print('error_mask', error_mask.shape)

        average_absolute_diff_after = np.abs(clean_data - dirty_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        average_absolute_diff_after = np.abs(clean_data - dirty_data).sum()
        print("平均修复距离：", average_absolute_diff_after)

class DataBase:
    def __init__(self, dataframe):
        self.dataframe = dataframe

class VariDataCleaner1_detection:
    @staticmethod
    def variance_constraint_clean(db: DataBase, variance_constraints: dict, ra: list, w_v=10, beta=0.5):
        
        error_mask = np.zeros(db.dataframe.shape, dtype=int)

        for attr in ra:
            assert attr in db.dataframe.columns
            assert attr in variance_constraints.keys()
            sequence = np.array(db.dataframe[attr])

            for k in range(w_v - 1, db.dataframe.shape[0] - 2 * w_v):
                for i in range(k - w_v + 1, k + 1):
                    
                    if np.var(sequence[i:i + w_v]) > variance_constraints[attr]:
                        
                        error_mask[k, db.dataframe.columns.get_loc(attr)] = 1
                    else:
                        error_mask[k, db.dataframe.columns.get_loc(attr)] = 0

        return error_mask

    @staticmethod
    def test_variance_constraint_clean():
  
        clean_data_df = pd.read_csv('idf.csv', header=None, low_memory=False).iloc[1:, :]
        clean_data = clean_data_df.values.astype(np.float64)


        error_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        dirty_data, clean_data = generate_dirty_data(clean_data, 0.1, error_types, covered_attrs=range(clean_data.shape[1]))


        average_absolute_diff_before = np.abs(clean_data - dirty_data).sum()

        attr_indices = range(clean_data.shape[1])
        variance_constraints = discover_variance_constraints(clean_data, attr_indices, w=10)


        db = DataBase(pd.DataFrame(dirty_data))
        error_mask = VariDataCleaner1.variance_constraint_clean(db, variance_constraints, attr_indices, w=10, beta=0.5)


        print("Error Mask:")
        print(error_mask)
        print("Error Mask Shape:", error_mask.shape)

        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        average_absolute_diff_after = np.abs(dirty_data - clean_data).sum()
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)


        average_absolute_diff_after_repair = np.abs(dirty_data - clean_data).sum()
        print("平均修复距离：", average_absolute_diff_after_repair)



class LOFDetection(BaseCleaningAlgorithm):
    """
    LOF (Local Outlier Factor) 异常检测方法
    使用局部离群因子检测异常点
    """
    def __init__(self, n_neighbors=20, contamination=0.1):
        """
        参数:
            n_neighbors: 邻居数量
            contamination: 异常值比例
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
    
    def clean(self, clean_data, current_dirty_data, **args):
        """
        使用LOF检测异常
        
        参数:
            clean_data: 干净数据（用于参考）
            current_dirty_data: 当前脏数据
            **args: 其他参数
        
        返回:
            error_mask: 错误掩码，1表示检测到错误
        """
        n_rows, n_cols = current_dirty_data.shape
        error_mask = np.zeros_like(current_dirty_data, dtype=int)
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(current_dirty_data)
        
        # 对每一列分别检测
        for col in range(n_cols):
            col_data = data_scaled[:, col].reshape(-1, 1)
            
            # 使用LOF检测
            lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, n_rows - 1),
                contamination=self.contamination
            )
            outlier_labels = lof.fit_predict(col_data)
            
            # -1表示异常点
            outlier_mask = outlier_labels == -1
            error_mask[outlier_mask, col] = 1
        
        return error_mask


class DBSCANDetection(BaseCleaningAlgorithm):
    """
    DBSCAN 密度聚类异常检测方法
    使用密度聚类识别异常点
    """
    def __init__(self, eps=0.5, min_samples=5):
        """
        参数:
            eps: 邻域半径
            min_samples: 最小样本数
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def clean(self, clean_data, current_dirty_data, **args):
        """
        使用DBSCAN检测异常
        
        参数:
            clean_data: 干净数据（用于参考）
            current_dirty_data: 当前脏数据
            **args: 其他参数
        
        返回:
            error_mask: 错误掩码，1表示检测到错误
        """
        n_rows, n_cols = current_dirty_data.shape
        error_mask = np.zeros_like(current_dirty_data, dtype=int)
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(current_dirty_data)
        
        # 对每一列分别检测
        for col in range(n_cols):
            col_data = data_scaled[:, col].reshape(-1, 1)
            
            # 使用DBSCAN聚类
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            cluster_labels = dbscan.fit_predict(col_data)
            
            # -1表示噪声点（异常点）
            outlier_mask = cluster_labels == -1
            error_mask[outlier_mask, col] = 1
        
        return error_mask


class ZScoreDetection(BaseCleaningAlgorithm):
    """
    Z-Score 异常检测方法
    使用标准差检测异常值
    """
    def __init__(self, threshold=3.0):
        """
        参数:
            threshold: Z-Score阈值，超过该值被视为异常
        """
        self.threshold = threshold
    
    def clean(self, clean_data, current_dirty_data, **args):
        """
        使用Z-Score检测异常
        
        参数:
            clean_data: 干净数据（用于参考）
            current_dirty_data: 当前脏数据
            **args: 其他参数
        
        返回:
            error_mask: 错误掩码，1表示检测到错误
        """
        n_rows, n_cols = current_dirty_data.shape
        error_mask = np.zeros_like(current_dirty_data, dtype=int)
        
        # 对每一列计算Z-Score
        for col in range(n_cols):
            col_data = current_dirty_data[:, col]
            mean = np.mean(col_data)
            std = np.std(col_data)
            
            if std == 0:
                continue
            
            # 计算Z-Score
            z_scores = np.abs((col_data - mean) / std)
            
            # 标记异常值
            outlier_mask = z_scores > self.threshold
            error_mask[outlier_mask, col] = 1
        
        return error_mask
