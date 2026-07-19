import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from constraints import RowConstraintMiner, RowConstraintMiner1, ColConstraintMiner, discover_variance_constraints
from single import MTSCleanRow, MTSCleanRow_c, MTSClean, LocalSpeedAccelClean, GlocalSpeedAccelClean, VarianceConstraintClean, VariDataCleaner1, LocalSpeedClean, GlobalSpeedClean, MovingAverageClean, WeightedMovingAverageClean, ExponentialWeightedMovingAverageClean
from dual_detection import MTSCleanRow_detection, LocalSpeedAccelClean_detection, GlocalSpeedAccelClean_detection, VarianceConstraintClean_detection, VariDataCleaner1_detection, LocalSpeedClean_detection, GlobalSpeedClean_detection, LOFDetection, DBSCANDetection, ZScoreDetection
from dual_repair import MTSCleanRow_repair, LocalSpeedAccelClean_repair, GlocalSpeedAccelClean_repair, VarianceConstraintClean_repair, VariDataCleaner1_repair, LocalSpeedClean_repair, GlobalSpeedClean_repair, LinearInterpolationRepair, KNNRepair, MultiColumnKNNRepair 
from scipy.optimize import linprog
from tqdm import tqdm


class DataBase:
    def __init__(self, dataframe):
        self.dataframe = dataframe


def generate_dirty_data(clean_data_w, error_ratio, error_types, covered_attrs):
    
    data = clean_data_w.copy()
    n_rows, n_cols = data.shape
    total_errors = int(n_rows * error_ratio)
    error_mask = np.zeros_like(data, dtype=int)
    
    while total_errors > 0:
        error_length = random.randint(10, 15)
        if total_errors < error_length:
            error_length = total_errors

        start_row = random.randint(10, n_rows - error_length)
        selected_cols = random.sample(list(covered_attrs), k=1)

        error_type = random.choice(error_types)
        print(error_type)
        if error_type == 'drift':
            for col in selected_cols:
                drift_value = 0.3
                if (data[start_row:start_row + error_length, col] + drift_value < 1).any():
                    print('add')
                    data[start_row:start_row + error_length, col] += drift_value
                    error_mask[start_row:start_row + error_length, col] = 1
        
        elif error_type == 'gaussian':
            snr = 20
            for col in selected_cols:
                signal_power = np.mean(data[:, col] ** 2)
                noise_power = signal_power / (10 ** (snr / 10))
                noise = np.random.normal(0, np.sqrt(noise_power), error_length)
                if (data[start_row:start_row + error_length, col] + noise < 1).any():
                    print('add')
                    data[start_row:start_row + error_length, col] += noise
                    error_mask[start_row:start_row + error_length, col] = 1

        elif error_type == 'volatility':
            for col in selected_cols:
                volatility_factors = np.random.uniform(0.9, 1.1, error_length)
                original_values = data[start_row:start_row + error_length, col]
                if (original_values * volatility_factors < 1).any:
                    print('add')
                    data[start_row:start_row + error_length, col] = original_values * volatility_factors
                    error_mask[start_row:start_row + error_length, col] = 1

        elif error_type == 'gradual':
            for col in selected_cols:
                direction = random.choice([-1, 1])
                magnitude = 0.2 * direction
                gradual_change = np.linspace(0, magnitude, error_length)
                gradual_change[-1] = 0
                if (data[start_row:start_row + error_length, col] + gradual_change < 1).any():
                    print('add')
                    data[start_row:start_row + error_length, col] += gradual_change
                    error_mask[start_row:start_row + error_length, col] = 1

        elif error_type == 'sudden':
            for col in selected_cols:
                direction = random.choice([-1, 1])
                magnitude = 0.2 * direction
                sudden_change = np.full(error_length, magnitude)
                recovery_length = error_length - (error_length // 2)
                sudden_change[-recovery_length:] = np.linspace(magnitude, 0, recovery_length)
                if (data[start_row:start_row + error_length, col] + sudden_change < 1).any():
                    print('add')
                    data[start_row:start_row + error_length, col] += sudden_change
                    error_mask[start_row:start_row + error_length, col] = 1

        total_errors -= error_length
    
    print(error_mask.sum(), error_mask.shape)
    dirty_data = data
    return dirty_data, clean_data_w


def data_cleaning(method, clean_data, current_dirty_data):
    if method == 'local_speed_clean':
        col_miner = ColConstraintMiner(clean_data)
        speed_constraints = col_miner.mine_col_constraints()[0]
        instance = LocalSpeedClean()
        cleaned_data = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints)
        return cleaned_data
    
    elif method == 'global_speed_clean':
        col_miner = ColConstraintMiner(clean_data)
        speed_constraints = col_miner.mine_col_constraints()[0]
        instance = GlobalSpeedClean()
        cleaned_data = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints)
        return cleaned_data

    elif method == 'local_speed_accel_clean':
        col_miner = ColConstraintMiner(clean_data)
        speed_constraints, accel_constraints = col_miner.mine_col_constraints()
        instance = LocalSpeedAccelClean()
        cleaned_data = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints,
                                      accel_constraints=accel_constraints)
        return cleaned_data

    elif method == 'global_speed_accel_clean':
        col_miner = ColConstraintMiner(clean_data)
        speed_constraints, accel_constraints = col_miner.mine_col_constraints()
        instance = GlocalSpeedAccelClean()
        cleaned_data = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints,
                                      accel_constraints=accel_constraints)
        return cleaned_data

    elif method == 'MTSCleanRow':
        row_miner = RowConstraintMiner(clean_data)
        row_constraints, covered_attrs = row_miner.mine_row_constraints()
        instance = MTSCleanRow()
        cleaned_data = instance.clean(clean_data, current_dirty_data, row_constraints=row_constraints)
        return cleaned_data

    elif method == 'VariDataCleaner1':
        attr_indices = range(clean_data.shape[1])
        variance_constraints = discover_variance_constraints(clean_data, attr_indices, w_v=10)
        instance = VariDataCleaner1()
        db = DataBase(pd.DataFrame(current_dirty_data))
        cleaned_data = instance.variance_constraint_clean(db, variance_constraints, attr_indices, w_v=10, beta=0.5)
        return cleaned_data

    elif method == 'MTSClean':
        row_miner = RowConstraintMiner(clean_data)
        row_constraints, covered_attrs = row_miner.mine_row_constraints(attr_num=3)
        
        col_miner = ColConstraintMiner(clean_data)
        speed_constraints, _ = col_miner.mine_col_constraints()
        
        instance = MTSClean()
        cleaned_data = instance.clean(clean_data, current_dirty_data, 
                                     row_constraints=row_constraints, 
                                     speed_constraints=speed_constraints)
        return cleaned_data
        
    elif method == 'moving_average_clean':
        instance = MovingAverageClean(window_size=5)
        cleaned_data = instance.clean(clean_data, current_dirty_data)
        return cleaned_data
    
    elif method == 'weighted_moving_average_clean':
        instance = WeightedMovingAverageClean(window_size=5)
        cleaned_data = instance.clean(clean_data, current_dirty_data)
        return cleaned_data
    
    elif method == 'exponential_weighted_ma_clean':
        instance = ExponentialWeightedMovingAverageClean(alpha=0.3)
        cleaned_data = instance.clean(clean_data, current_dirty_data)
        return cleaned_data


class DataCleaningEnv():
    def __init__(self, clean_data, window_size, direct_dirty_data=None):
        super(DataCleaningEnv, self).__init__()
        self.window_size = window_size
        self.clean_data = clean_data
        self.direct_dirty_data = direct_dirty_data
        self.data_types = ['drift', 'gaussian', 'volatility', 'gradual', 'sudden']
        self.cleaning_methods = ['local_speed_clean', 'global_speed_clean', 'local_speed_accel_clean', 'MTSCleanRow', 'global_speed_accel_clean', 'VariDataCleaner1', 'MTSClean', 'moving_average_clean', 'weighted_moving_average_clean', 'exponential_weighted_ma_clean']
        self.action_space = len(self.cleaning_methods)
        self.detection_methods = ['local_speed_clean_detection', 'global_speed_clean_detection', 'local_speed_accel_clean_detection', 'global_speed_accel_clean_detection', 'MTSCleanRow_detection', 'VariDataCleaner1_detection', 'lof_detection', 'dbscan_detection', 'zscore_detection']
        self.repair_methods = ['local_speed_clean_repair', 'global_speed_clean_repair', 'local_speed_accel_clean_repair', 'global_speed_accel_clean_repair', 'MTSCleanRow_repair', 'VariDataCleaner1_repair', 'linear_interpolation_repair', 'knn_repair', 'multi_column_knn_repair']
        self.detection_action_space = len(self.detection_methods)
        self.repair_action_space = len(self.repair_methods)
        self.clean_data_window = None
        self.previous_cleaned_data = None  # 保留作为后备
      
        self.history_cleaned_windows = []  # 存储 (window_index, cleaned_data, metadata)
        self.max_history_size = 100  # 最大历史记录数量
        
        
        self.similarity_threshold = 0.85  
        self.adaptive_threshold_enabled = True  
        self.threshold_adjustment_rate = 0.05  
        self.threshold_min = 0.6  
        self.threshold_max = 0.95 
        self.reward_history = deque(maxlen=20)  # 存储最近的奖励用于评估
        self.threshold_adjustment_interval = 5  
        self.window_counter = 0  
        
        
        # 存储直接清洗和检测+修复路径的原始reward历史，用于在线归一化
        self.direct_clean_reward_history = deque(maxlen=100)  # 存储直接清洗reward
        self.detection_repair_reward_history = deque(maxlen=100)  # 存储检测+修复reward
        # ==============================================

    def _calculate_similarity(self, data1, data2):
        
        # 确保数据维度一致
        if data1.shape != data2.shape:
            
            min_rows = min(data1.shape[0], data2.shape[0])
            min_cols = min(data1.shape[1], data2.shape[1])
            data1_aligned = data1[:min_rows, :min_cols]
            data2_aligned = data2[:min_rows, :min_cols]
        else:
            data1_aligned = data1
            data2_aligned = data2
        
        
        flat1 = data1_aligned.flatten()
        flat2 = data2_aligned.flatten()
        
        
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(flat1, flat2) / (norm1 * norm2)
        
        
        similarity = (cosine_sim + 1) / 2
        
        return similarity

    def _find_most_similar_history(self, current_data, window_index):
        """
        从历史修复窗口中查找与当前数据最相似的历史窗口
        
        """
        if len(self.history_cleaned_windows) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for history_entry in self.history_cleaned_windows:
            hist_idx = history_entry['window_index']
            hist_cleaned_data = history_entry['cleaned_data']
            
            
            if hist_idx == window_index:
                continue
            
            # 计算相似度
            similarity = self._calculate_similarity(current_data, hist_cleaned_data)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = hist_cleaned_data
        
        return best_match, best_similarity

    def _add_to_history(self, window_index, cleaned_data, metadata=None):
        """
        将清洗后的窗口数据添加到历史记录中
        
        
        """
        if metadata is None:
            metadata = {}
        
        history_entry = {
            'window_index': window_index,
            'cleaned_data': cleaned_data.copy(),
            'metadata': metadata,
            'timestamp': len(self.history_cleaned_windows)  
        }
        
        self.history_cleaned_windows.append(history_entry)
        
        # 保持历史记录大小限制
        if len(self.history_cleaned_windows) > self.max_history_size:
            self.history_cleaned_windows.pop(0)

    def _adjust_threshold_adaptively(self, recent_rewards):
       
        if not self.adaptive_threshold_enabled or len(recent_rewards) < 5:
            return
        
        # 计算奖励的均值和标准差
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
       
        if avg_reward > 0.7:  
            adjustment = self.threshold_adjustment_rate
            self.similarity_threshold = min(self.threshold_max, 
                                            self.similarity_threshold + adjustment)
            print(f"Performance good (avg reward: {avg_reward:.3f}), increasing threshold to {self.similarity_threshold:.3f}")
        
        elif avg_reward < 0.3:  
            adjustment = self.threshold_adjustment_rate
            self.similarity_threshold = max(self.threshold_min, 
                                            self.similarity_threshold - adjustment)
            print(f"Performance poor (avg reward: {avg_reward:.3f}), decreasing threshold to {self.similarity_threshold:.3f}")
        
        else:  
            if reward_std < 0.1: 
                self.similarity_threshold = min(self.threshold_max, 
                                                self.similarity_threshold + self.threshold_adjustment_rate * 0.5)
                print(f"Stable performance, fine-tuning threshold to {self.similarity_threshold:.3f}")

    def _update_threshold_with_reward(self, reward):
        
        self.reward_history.append(reward)
        self.window_counter += 1
        
       
        if self.window_counter % self.threshold_adjustment_interval == 0:
            self._adjust_threshold_adaptively(list(self.reward_history))

    def _get_reference_data(self, current_data, window_index, use_similarity=True):
        """
        获取参考数据（用于计算奖励）
        
        """
        if use_similarity and len(self.history_cleaned_windows) > 0:
            # 使用相似性匹配查找最相似的历史窗口
            best_match, similarity = self._find_most_similar_history(current_data, window_index)
            
           
            if best_match is not None and similarity >= self.similarity_threshold:
                print(f"Using similarity match reference (similarity: {similarity:.4f}, threshold: {self.similarity_threshold:.4f})")
                return best_match, 'similarity_match'
            elif best_match is not None:
                print(f"Similarity ({similarity:.4f}) below threshold ({self.similarity_threshold:.4f}), falling back")
                
                if self.previous_cleaned_data is not None:
                    return self.previous_cleaned_data, 'previous_window'
        
        # 使用上一窗口的清洗数据
        if self.previous_cleaned_data is not None:
            print("Using previous window reference")
            return self.previous_cleaned_data, 'previous_window'
        
        # 如果都没有，使用同源历史数据
        print("Using clean data reference (first window)")
        return self.clean_data_window, 'clean_data'

    def _normalize_reward(self, reward, reward_history, epsilon=1e-8):
        """
        使用历史数据对reward进行归一化（Min-Max归一化）
        
        """
        if len(reward_history) == 0:
            # 没有历史数据时，使用sigmoid函数进行软归一化
            
            normalized = 1.0 / (1.0 + np.exp(-reward))
            return normalized
        
        # 计算历史数据的最小值和最大值
        history_array = np.array(reward_history)
        min_val = np.min(history_array)
        max_val = np.max(history_array)
        
        # 如果范围太小，使用sigmoid归一化作为备选
        if max_val - min_val < epsilon:
            normalized = 1.0 / (1.0 + np.exp(-reward))
            return normalized
        
        # Min-Max归一化到[0, 1]
        normalized = (reward - min_val) / (max_val - min_val)
        
        # 限制范围
        normalized = max(0.0, min(1.0, normalized))
        
        return normalized
    
    def _get_normalized_direct_clean_reward(self, raw_reward):
        """
        获取归一化的直接清洗reward
        
        """
        # 将当前reward添加到历史中（用于后续归一化）
        normalized = self._normalize_reward(raw_reward, self.direct_clean_reward_history)
        
        return normalized
    
    def _get_normalized_detection_repair_reward(self, detection_reward, repair_reward):
        """
        获取归一化的检测+修复综合reward
        
        """
       
        # raw_combined = repair_reward
        raw_combined = detection_reward + repair_reward
        
        normalized = self._normalize_reward(raw_combined, self.detection_repair_reward_history)
        
        return normalized
    
    def _update_reward_histories(self, direct_reward, detection_reward, repair_reward):
        """
        更新reward历史记录（在比较后调用）
        
        """
        self.direct_clean_reward_history.append(direct_reward)
        
        # detection_repair_combined = repair_reward
        detection_repair_combined = detection_reward + repair_reward
        self.detection_repair_reward_history.append(detection_repair_combined)
    # ======================================

    def reset(self, window_index):
        current_data_types = self.data_types
        self.current_dirty_data, self.clean_data_window = generate_dirty_data(
            self.clean_data[window_index * self.window_size: window_index * self.window_size + self.window_size, :],
            0.1,
            current_data_types,
            covered_attrs=range(self.clean_data.shape[1]))
        
        # 获取参考数据（使用相似性匹配）
        self.reference_data, self.reference_type = self._get_reference_data(
            self.current_dirty_data, window_index, use_similarity=True
        )
            
        return self.current_dirty_data, self.clean_data_window

    def reset_with_only_window(self, window_index):
        self.direct_dirty_data_window = self.direct_dirty_data[window_index * self.window_size: window_index * self.window_size + self.window_size, :]
        self.clean_data_window = self.clean_data[window_index * self.window_size: window_index * self.window_size + self.window_size, :]
        
        # 获取参考数据（使用相似性匹配）
        self.reference_data, self.reference_type = self._get_reference_data(
            self.direct_dirty_data_window, window_index, use_similarity=True
        )
            
        return self.direct_dirty_data_window, self.clean_data_window

   
    def step(self, action, window_index, clean_data, state):
        cleaning_method = self.cleaning_methods[action]
        cleaned_data = data_cleaning(cleaning_method, clean_data, state)
        
        # 使用参考数据计算奖励
        diff_state_to_ref = np.abs(self.reference_data - state).sum()
        diff_cleaned_to_ref = np.abs(self.reference_data - cleaned_data).sum()

        difference = diff_cleaned_to_ref - diff_state_to_ref

        if difference < 0:
            reward = np.log(-difference + 1)  # Positive reward for improvement
        elif difference > 0:
            reward = -np.log(difference + 1)  # Negative reward for worsening
        else:
            reward = 0  # Neutral reward for no change

      
        self._update_threshold_with_reward(reward)

        if diff_cleaned_to_ref < 0.01:
            done = True
            # 存储清洗后的数据到历史记录
            metadata = {
                'cleaning_method': cleaning_method,
                'action': action,
                'reference_type': self.reference_type,
                'window_index': window_index,
                'raw_reward': reward  # 记录原始reward
            }
            self._add_to_history(window_index, cleaned_data, metadata)
            # 同时更新上一窗口数据作为后备
            self.previous_cleaned_data = cleaned_data
        else:
            done = False

        print(f"Window {window_index} - Action: {action} (Reference: {self.reference_type})")
        print(f"Difference Sum between State and Reference Data: {diff_state_to_ref}")
        print(f"Difference Sum between Cleaned Data and Reference Data: {diff_cleaned_to_ref}")
        print(f"Raw Reward: {reward}")

        return cleaned_data, reward, done, None
    # ========================================================

    def detection_step(self, action, window_index, clean_data, state):
        detection_method = self.detection_methods[action]
        detected_errors = self.apply_detection(detection_method, self.clean_data_window, state)
        
        # 计算检测步骤的原始reward (F1分数)
        detection_reward = self.calculate_detection_reward(detected_errors, state)
        
        print(f"Window {window_index} - Detection Action: {action}")
        print(f"Detection Raw Reward (F1): {detection_reward}")

        diff_state_to_ref = np.abs(self.reference_data - state).sum()
        print(f"Difference Sum between State and Reference Data: {diff_state_to_ref}")
        
        # 返回检测结果和原始reward
        return detected_errors, detection_reward

    def repair_step(self, action, window_index, clean_data, state, detected_errors):
        repair_method = self.repair_methods[action]
        repaired_data = self.apply_repair(repair_method, clean_data, state, detected_errors)
        
        # 使用参考数据计算修复步骤的原始reward
        diff_state_to_ref = np.abs(self.reference_data - state).sum()
        diff_cleaned_to_ref = np.abs(self.reference_data - repaired_data).sum()

        difference = diff_cleaned_to_ref - diff_state_to_ref

        if difference < 0:
            repair_reward = np.log(-difference + 1)  # Positive reward for improvement
        elif difference > 0:
            repair_reward = -np.log(difference + 1)  # Negative reward for worsening
        else:
            repair_reward = 0  # Neutral reward for no change

        print(f"Window {window_index} - Repair Action: {action} (Reference: {self.reference_type})")
        print(f"Difference Sum between State and Reference Data: {diff_state_to_ref}")
        print(f"Difference Sum between Cleaned Data and Reference Data: {diff_cleaned_to_ref}")
        print(f"Repair Raw Reward: {repair_reward}")

        if diff_cleaned_to_ref < 0.01:
            done = True
            # 存储修复后的数据到历史记录
            metadata = {
                'repair_method': repair_method,
                'detection_method': self.detection_methods[action] if action < len(self.detection_methods) else 'unknown',
                'action': action,
                'reference_type': self.reference_type,
                'window_index': window_index,
                'repair_raw_reward': repair_reward
            }
            self._add_to_history(window_index, repaired_data, metadata)
            # 同时更新上一窗口数据作为后备
            self.previous_cleaned_data = repaired_data
        else:
            done = False

        # 返回原始reward
        return repaired_data, repair_reward, done, None

    def apply_detection(self, method, clean_data, current_dirty_data):
        if method == 'local_speed_clean_detection':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints = col_miner.mine_col_constraints()[0]
            instance = LocalSpeedClean_detection()
            error_mask = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints)
        
        elif method == 'global_speed_clean_detection':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints = col_miner.mine_col_constraints()[0]
            instance = GlobalSpeedClean_detection()
            error_mask = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints)

        elif method == 'local_speed_accel_clean_detection':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints, accel_constraints = col_miner.mine_col_constraints()
            instance = LocalSpeedAccelClean_detection()
            error_mask = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints, accel_constraints=accel_constraints)

        elif method == 'global_speed_accel_clean_detection':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints, accel_constraints = col_miner.mine_col_constraints()
            instance = GlocalSpeedAccelClean_detection()
            error_mask = instance.clean(clean_data, current_dirty_data, speed_constraints=speed_constraints, accel_constraints=accel_constraints)
    
        elif method == 'MTSCleanRow_detection':
            row_miner = RowConstraintMiner(clean_data)
            row_constraints, covered_attrs = row_miner.mine_row_constraints()
            instance = MTSCleanRow_detection()
            error_mask = instance.clean(clean_data, current_dirty_data, row_constraints=row_constraints)
        
        elif method == 'VariDataCleaner1_detection':
            attr_indices = range(clean_data.shape[1])
            variance_constraints = discover_variance_constraints(clean_data, attr_indices, w_v=10)
            instance = VariDataCleaner1_detection()
            db = DataBase(pd.DataFrame(current_dirty_data))
            error_mask = instance.variance_constraint_clean(db, variance_constraints, attr_indices, w_v=10, beta=0.5)

        elif method == 'lof_detection':
            instance = LOFDetection(n_neighbors=20, contamination=0.1)
            error_mask = instance.clean(clean_data, current_dirty_data)
    
        elif method == 'dbscan_detection':
            instance = DBSCANDetection(eps=0.5, min_samples=5)
            error_mask = instance.clean(clean_data, current_dirty_data)
        
        elif method == 'zscore_detection':
            instance = ZScoreDetection(threshold=3.0)
            error_mask = instance.clean(clean_data, current_dirty_data)
            
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        return error_mask
    
    def calculate_detection_reward(self, detected_errors, dirty_data_window):
        """
        计算检测步骤的奖励，使用参考数据作为真实值
        
        """
        # 使用参考数据作为真实值
        true_errors = self.reference_data != dirty_data_window
        
        true_positives = np.sum(detected_errors & true_errors)
        false_positives = np.sum(detected_errors & ~true_errors)
        false_negatives = np.sum(~detected_errors & true_errors)
        
        # 计算精确率和召回率
        precision = true_positives / (false_positives + true_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (false_negatives + true_positives) if (true_positives + false_negatives) > 0 else 0
        
        # 使用F1分数作为奖励
        reward = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Detection Reward - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {reward:.4f}")
        
        return reward

    def apply_repair(self, method, clean_data, current_dirty_data, error_mask):
        if method == 'local_speed_clean_repair':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints = col_miner.mine_col_constraints()[0]
            instance = LocalSpeedClean_repair()
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask, speed_constraints=speed_constraints)
        
        elif method == 'global_speed_clean_repair':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints = col_miner.mine_col_constraints()[0]
            instance = GlobalSpeedClean_repair()
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask, speed_constraints=speed_constraints)

        elif method == 'local_speed_accel_clean_repair':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints, accel_constraints = col_miner.mine_col_constraints()
            instance = LocalSpeedAccelClean_repair()
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask, speed_constraints=speed_constraints, accel_constraints=accel_constraints)

        elif method == 'global_speed_accel_clean_repair':
            col_miner = ColConstraintMiner(clean_data)
            speed_constraints, accel_constraints = col_miner.mine_col_constraints()
            instance = GlocalSpeedAccelClean_repair()
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask, speed_constraints=speed_constraints, accel_constraints=accel_constraints)
    
        elif method == 'MTSCleanRow_repair':
            row_miner = RowConstraintMiner(clean_data)
            row_constraints, covered_attrs = row_miner.mine_row_constraints()
            instance = MTSCleanRow_repair()
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask, row_constraints=row_constraints)
        
        elif method == 'VariDataCleaner1_repair':
            attr_indices = range(clean_data.shape[1])
            variance_constraints = discover_variance_constraints(clean_data, attr_indices, w_v=10)
            instance = VariDataCleaner1_repair()
            db = DataBase(pd.DataFrame(current_dirty_data))
            repaired_data = instance.variance_constraint_clean(db, variance_constraints, attr_indices, error_mask, w_v=10, beta=0.5)

        elif method == 'linear_interpolation_repair':
            instance = LinearInterpolationRepair()
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask)
        
        elif method == 'knn_repair':
            instance = KNNRepair(n_neighbors=5)
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask)
        
        elif method == 'multi_column_knn_repair':
            instance = MultiColumnKNNRepair(n_neighbors=5)
            repaired_data = instance.clean(clean_data, current_dirty_data, error_mask)
            
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        return repaired_data

    def calculate_repair_reward(self, repaired_data, clean_data_window, dirty_data_window, epsilon=0.1):
        diff_state_to_ref = np.abs(self.reference_data - dirty_data_window).sum()
        diff_cleaned_to_ref = np.abs(self.reference_data - repaired_data).sum()

        difference = diff_cleaned_to_ref - diff_state_to_ref

        if difference < 0:
            reward = np.log(-difference + 1)
        elif difference > 0:
            reward = -np.log(difference + 1)
        else:
            reward = 0

        print(f"Repair Reward: {reward}")
        return reward

    def check_termination(self, repaired_data, clean_data_window):
        return np.abs(repaired_data - self.reference_data).sum() < 0.01

    def clear_history(self):
        """清空历史记录"""
        self.history_cleaned_windows = []
        self.previous_cleaned_data = None
        print("History cleared")

    def get_history_stats(self):
        """获取历史记录的统计信息"""
        if len(self.history_cleaned_windows) == 0:
            return {"count": 0, "avg_similarity": 0.0}
        
        return {
            "count": len(self.history_cleaned_windows),
            "max_size": self.max_history_size,
            "similarity_threshold": self.similarity_threshold,
            "adaptive_enabled": self.adaptive_threshold_enabled,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max
        }
    
    def configure_adaptive_threshold(self, enabled=True, initial_threshold=0.85, 
                                     min_threshold=0.6, max_threshold=0.95, 
                                     adjustment_rate=0.05, adjustment_interval=5):
        
        self.adaptive_threshold_enabled = enabled
        self.similarity_threshold = initial_threshold
        self.threshold_min = min_threshold
        self.threshold_max = max_threshold
        self.threshold_adjustment_rate = adjustment_rate
        self.threshold_adjustment_interval = adjustment_interval
        print(f"Adaptive threshold configured: enabled={enabled}, threshold={initial_threshold}")
