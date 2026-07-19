import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from env import DataCleaningEnv
from model import ActorCritic
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from natsort import natsorted
import csv
import os
import time
import copy
from torch.optim import Adam


# def set_seed(seed=42):
#     """固定所有随机种子"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     print(f"Random seed {seed} has been set")


# set_seed(42)
# # ==========================================

# 参数设置
num_episodes = 30  # 总迭代次数
meta_epochs = 4    # 元学习迭代次数
num_tasks = 4       # 任务数量（数据集数量）
inner_epochs = 1    # 内部更新次数
gamma = 0.9         # 折扣因子
actor_lr = 1e-3     # 策略网络的学习率
critic_lr = 1e-2    # 价值网络的学习率
meta_lr = 1e-3      # 元学习率
n_hiddens = 64      # 隐含层神经元个数
detection_return_list = []  # 保存每个回合的return
repair_return_list = []
direct_clean_return_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 存储直接清洗和检测+修复路径的原始reward历史，用于在线归一化
global_direct_clean_reward_history = deque(maxlen=200)
global_detection_repair_reward_history = deque(maxlen=200)


def normalize_reward(reward, reward_history, epsilon=1e-8):
    """
    使用历史数据对reward进行归一化（Min-Max归一化）
    
    参数:
        reward: 原始reward值
        reward_history: 历史reward列表（deque）
        epsilon: 防止除零的小常数
        
    返回:
        normalized_reward: 归一化后的reward值（范围[0,1]）
    """
    if len(reward_history) == 0:
        # 没有历史数据时，使用sigmoid函数进行软归一化
        
        normalized = 1.0 / (1.0 + np.exp(-reward / 2.0))
        return normalized
    
    # 计算历史数据的最小值和最大值
    history_array = np.array(reward_history)
    min_val = np.min(history_array)
    max_val = np.max(history_array)
    
    # 如果范围太小，使用sigmoid归一化作为备选
    if max_val - min_val < epsilon:
        normalized = 1.0 / (1.0 + np.exp(-reward / 2.0))
        return normalized
    
    # Min-Max归一化到[0, 1]
    normalized = (reward - min_val) / (max_val - min_val)
    
    # 限制范围
    normalized = max(0.0, min(1.0, normalized))
    
    return normalized


def get_normalized_direct_clean_reward(raw_reward):
    """获取归一化的直接清洗reward"""
    return normalize_reward(raw_reward, global_direct_clean_reward_history)


def get_normalized_detection_repair_reward(detection_reward, repair_reward):
    """获取归一化的检测+修复综合reward"""
    # raw_combined = repair_reward
    raw_combined = detection_reward + repair_reward
    return normalize_reward(raw_combined, global_detection_repair_reward_history)


def update_reward_histories(direct_reward, detection_reward, repair_reward):
    """更新reward历史记录（在比较后调用）"""
    global global_direct_clean_reward_history, global_detection_repair_reward_history
    global_direct_clean_reward_history.append(direct_reward)
    # detection_repair_combined = repair_reward
    detection_repair_combined = detection_reward + repair_reward
    global_detection_repair_reward_history.append(detection_repair_combined)
# =========================================================

# 环境加载
train_datasets = {
    'Energy': 'Energy_Clean.csv',
    'Oil': 'IDF_OilTemp_Clean.csv',
    'Power': 'IDF_Power_Clean.csv',
    'IDF': 'idf.csv'
}

# 测试数据集
test_dirty_dataset = 'Energy_Dirty.csv'
test_clean_dataset = 'Energy_Clean.csv'
# test_dirty_dataset = 'IDF_Power_Dirty.csv'
# test_clean_dataset = 'IDF_Power_Clean.csv'
# test_dirty_dataset = 'IDF_OilTemp_Dirty.csv'
# test_clean_dataset = 'IDF_OilTemp_Clean.csv'
# test_dirty_dataset = 'idf_2.pth.01.csv'
# test_clean_dataset = 'idf.csv'
# test_dirty_dataset = 'IDF_OilTemp_Dirty_14.pth.01_10000.csv'
# test_clean_dataset = 'IDF_OilTemp_Clean.csv'
# test_dirty_dataset = 'IDF_Power_Dirty_28.pth.01.csv'
# test_clean_dataset = 'IDF_Power_Clean.csv'


# 初始化环境和数据
envs = {}
scalers = {}
clean_datas = {}
num_columns = {}
window_sizes = 200  # 统一窗口大小

# 加载训练数据集
for name, path in train_datasets.items():
    clean_data_df = pd.read_csv(path, header=None, low_memory=False).iloc[1:, 1:]
    num_column = clean_data_df.shape[1]
    clean_data = np.array(clean_data_df.values)
    clean_data = clean_data.astype(np.float64)
    scaler = MinMaxScaler()
    clean_data = scaler.fit_transform(clean_data)
    
    envs[name] = DataCleaningEnv(clean_data, window_size=window_sizes)
    scalers[name] = scaler
    clean_datas[name] = clean_data
    num_columns[name] = num_column

# 模型构建
def create_agent(env, num_column):
    """Create agents with specific input dimensions"""
    detection_agent = ActorCritic(num_column=num_column,
                    window_size=window_sizes,
                    n_hiddens=n_hiddens,
                    n_actions=env.detection_action_space,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    device=device).to(device)
    
    repair_agent = ActorCritic(num_column=num_column,
                    window_size=window_sizes,
                    n_hiddens=n_hiddens,
                    n_actions=env.repair_action_space,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    device=device).to(device)
    
    direct_clean_agent = ActorCritic(num_column=num_column,
                    window_size=window_sizes,
                    n_hiddens=n_hiddens,
                    n_actions=env.action_space,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    device=device).to(device)
    
    return detection_agent, repair_agent, direct_clean_agent

def load_compatible_weights(target_model, source_state_dict):
    """Load only compatible weights between models of different dimensions"""
    target_state_dict = target_model.state_dict()
    
    # 获取输入和输出维度信息
    target_input_dim = target_model.num_column
    source_input_dim = source_state_dict['actor.tcn.network.0.conv1.weight_v'].shape[1]
    
    # 获取输出维度信息
    target_output_dim = target_state_dict['actor.tcn.network.0.conv1.weight_v'].shape[0]
    source_output_dim = source_state_dict['actor.tcn.network.0.conv1.weight_v'].shape[0]
    
    # 只加载兼容的权重
    compatible_state_dict = {}
    for k, v in source_state_dict.items():
        if k in target_state_dict:
            # 处理输入层权重
            if 'conv1.weight_v' in k or 'net.0.weight_v' in k:
                # 调整输入和输出维度
                if v.shape[1] == source_input_dim and target_input_dim <= v.shape[1]:
                    # 截取匹配的维度部分
                    if v.shape[0] == source_output_dim and target_output_dim <= v.shape[0]:
                        # 同时截取输入和输出维度
                        compatible_state_dict[k] = v[:target_output_dim, :target_input_dim, ...].clone()
                    else:
                        # 只截取输入维度
                        compatible_state_dict[k] = v[:, :target_input_dim, ...].clone()
            # 处理其他层权重
            elif v.shape == target_state_dict[k].shape:
                compatible_state_dict[k] = v
    
    target_state_dict.update(compatible_state_dict)
    target_model.load_state_dict(target_state_dict, strict=False)
    return target_model

# 保存数据的文件
reward_csv_path = "comprehensive_episode_rewards_ene_real_fur_meta.csv"
loss_csv_path = "comprehensive_loss_ene_real_fur_meta.csv"

for csv_path, header in zip([reward_csv_path, loss_csv_path],
                            [["Episode", "Total Detection Reward", "Total Repair Reward", "Total Direct Clean Reward"], 
                             ["Episode", "detection_actor_loss", "detection_critic_loss", 
                              "repair_actor_loss", "repair_critic_loss", 
                              "direct_clean_actor_loss", "direct_clean_critic_loss"]]):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

# 初始化脏数据和清洗数据的文件夹
os.makedirs("comprehensive_generated_dirty_data_ene_real_fur_meta", exist_ok=True)
os.makedirs("comprehensive_cleaned_data_results_ene_real_fur_meta", exist_ok=True)

# 评测函数
def mse(d_repair: np.ndarray, d_clean: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        sum += mean_squared_error(data_clean_scaled[:, i], data_repair_scaled[:, i])
    return sum / dim

def mnad(d_repair: np.ndarray, d_clean: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        sum += mean_absolute_error(data_clean_scaled[:, i], data_repair_scaled[:, i])
    return sum / dim

def rra(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_dirty_scaled = scaler.transform(d_dirty)
    data_repair_scaled = scaler.transform(d_repair)
    mae_rc = mnad(d_repair, d_clean)
    mae_rd = 0
    mae_cd = mnad(d_dirty, d_clean)
    dim = data_repair_scaled.shape[1]
    for i in range(dim):
        mae_rd += mean_absolute_error(data_dirty_scaled[:, i], data_repair_scaled[:, i])
    mae_rd /= dim
    return 1 - mae_rc / (mae_cd + mae_rd)

def precision_recall(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    bound = (np.max(d_dirty, axis=0) - np.min(d_dirty, axis=0)) / 10
    error_array = np.abs(d_dirty - d_clean) > bound
    repair_array = np.abs(d_dirty - d_repair) > bound
    benefit_repair_array = (np.abs(d_clean - d_repair) < np.abs(d_dirty - d_clean)) * repair_array
    tp = error_array * benefit_repair_array
    P = np.sum(tp != 0) / (np.sum(repair_array != 0))
    R = np.sum(tp != 0) / (np.sum(error_array != 0))
    F1 = 2 * P * R / (P + R)
    return P, R, F1

def mae_error(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    d_repair = np.where(d_dirty == d_clean, d_clean, d_repair)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        sum += mean_absolute_error(data_clean_scaled[:, i], data_repair_scaled[:, i])
    return sum / dim

def meta_train():
    """元训练过程"""
    model_para_dir = "comprehensive_model_para_ene_real_fur_meta"
    os.makedirs(model_para_dir, exist_ok=True)
    
    # 创建元模型时使用测试数据集的维度
    meta_detection_agent, meta_repair_agent, meta_direct_clean_agent = create_agent(envs[name], num_columns[name])
    
    # 元优化器
    meta_detection_optimizer = Adam(meta_detection_agent.parameters(), lr=meta_lr)
    meta_repair_optimizer = Adam(meta_repair_agent.parameters(), lr=meta_lr)
    meta_direct_clean_optimizer = Adam(meta_direct_clean_agent.parameters(), lr=meta_lr)
    
    for epoch in range(meta_epochs):
        print(f"Meta Epoch {epoch + 1}/{meta_epochs}")
    
        # 保存原始参数
        original_detection_state = copy.deepcopy(meta_detection_agent.state_dict())
        original_repair_state = copy.deepcopy(meta_repair_agent.state_dict())
        original_direct_clean_state = copy.deepcopy(meta_direct_clean_agent.state_dict())
        
        total_detection_reward = 0
        total_repair_reward = 0
        total_direct_clean_reward = 0
        
        task_names = [name for name in envs.keys() if name != 'Test']  # 排除测试数据集
        random.shuffle(task_names)  # 随机打乱任务顺序

        # 在每个任务上进行训练
        for task_name, env in envs.items():
            if task_name == 'Test':  # 跳过测试数据集
                continue
                
            print(f"Training on task: {task_name}")
            
            # 创建任务特定模型（使用当前数据集的列数）
            task_detection_agent = ActorCritic(
                num_column=num_columns[task_name],
                window_size=window_sizes,
                n_hiddens=n_hiddens,
                n_actions=env.detection_action_space,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                device=device
            ).to(device)
            
            # 只加载兼容的权重
            task_detection_agent = load_compatible_weights(task_detection_agent, original_detection_state)
            
            task_repair_agent = ActorCritic(
                num_column=num_columns[task_name],
                window_size=window_sizes,
                n_hiddens=n_hiddens,
                n_actions=env.repair_action_space,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                device=device
            ).to(device)
            task_repair_agent = load_compatible_weights(task_repair_agent, original_repair_state)
            
            task_direct_clean_agent = ActorCritic(
                num_column=num_columns[task_name],
                window_size=window_sizes,
                n_hiddens=n_hiddens,
                n_actions=env.action_space,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                device=device
            ).to(device)
            task_direct_clean_agent = load_compatible_weights(task_direct_clean_agent, original_direct_clean_state)
            
            # 内部循环更新
            for inner_epoch in range(inner_epochs):
                results = train_on_task(
                    (task_detection_agent, task_repair_agent, task_direct_clean_agent),
                    env,
                    clean_datas[task_name],
                    int(clean_datas[task_name].shape[0] / window_sizes))
                
                total_detection_reward += results[0]
                total_repair_reward += results[1]
                total_direct_clean_reward += results[2]
                
                # 记录损失
                with open(loss_csv_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        epoch * (len(envs) - 1) + list(envs.keys()).index(task_name),
                        results[3], results[4], results[5], results[6], results[7], results[8]
                    ])
            
            # 计算梯度并更新元模型
            meta_detection_optimizer.zero_grad()
            meta_repair_optimizer.zero_grad()
            meta_direct_clean_optimizer.zero_grad()
            
            # 计算梯度
            for meta_param, task_param in zip(meta_detection_agent.parameters(), task_detection_agent.parameters()):
                if meta_param.shape == task_param.shape:
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param)
                    meta_param.grad.data.add_(meta_param.data - task_param.data)
                
            for meta_param, task_param in zip(meta_repair_agent.parameters(), task_repair_agent.parameters()):
                if meta_param.shape == task_param.shape:
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param)
                    meta_param.grad.data.add_(meta_param.data - task_param.data)
            
            for meta_param, task_param in zip(meta_direct_clean_agent.parameters(), task_direct_clean_agent.parameters()):
                if meta_param.shape == task_param.shape:
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param)
                    meta_param.grad.data.add_(meta_param.data - task_param.data)
        
        # 更新元参数
        meta_detection_optimizer.step()
        meta_repair_optimizer.step()
        meta_direct_clean_optimizer.step()
        
        # 保存模型参数
        torch.save(meta_detection_agent.state_dict(), 
                  os.path.join(model_para_dir, f"detection_actor_parameters_epoch_{epoch}.pth"))
        torch.save(meta_repair_agent.state_dict(), 
                 os.path.join(model_para_dir, f"repair_actor_parameters_epoch_{epoch}.pth"))
        torch.save(meta_direct_clean_agent.state_dict(),
                 os.path.join(model_para_dir, f"direct_clean_actor_parameters_epoch_{epoch}.pth"))

def train_on_task(agents, env, clean_data, num_windows, adapt=False):
    """在单个任务上训练"""
    detection_agent, repair_agent, direct_clean_agent = agents

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    for i in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** i))

        # 构造数据集，保存每个回合的状态数据
        detection_transition_dict = {
            'detection_states': [],
            'detection_actions': [],
            'detection_next_states': [],
            'detection_rewards': [],
            'detection_dones': []
        }

        repair_transition_dict = {
            'repair_states': [],
            'repair_actions': [],
            'repair_next_states': [],
            'repair_rewards': [],
            'repair_dones': []
        }
        
        direct_clean_transition_dict = {
            'direct_clean_states': [],
            'direct_clean_actions': [],
            'direct_clean_next_states': [],
            'direct_clean_rewards': [],
            'direct_clean_dones': []
        }
        
        total_detection_return = 0
        total_repair_return = 0
        total_direct_clean_return = 0
        w = 0
        
        while w < num_windows:
            state, clean_data_window = env.reset(window_index=w)
            done = False
            detection_episode_return = 0
            repair_episode_return = 0
            direct_clean_episode_return = 0
            step_counter = 0
            
            while not done and step_counter <= 5:
                # 尝试两种清洗策略并比较reward
                # 策略1: 直接清洗
                use_random = random.random() < epsilon
                
                if use_random:
                    direct_clean_action = np.random.choice(env.action_space)
                else:
                    direct_clean_action = direct_clean_agent.take_action(state)
                direct_clean_next_state, direct_clean_reward, direct_clean_done, _ = env.step(direct_clean_action, window_index=w, clean_data=clean_data, state=state)
                
                # 策略2: 检测+修复
                if use_random:
                    detection_action = np.random.choice(env.detection_action_space)
                else:
                    detection_action = detection_agent.take_action(state)
                detected_errors, detection_reward = env.detection_step(detection_action, window_index=w, clean_data=clean_data, state=state)
                
                if use_random:
                    repair_action = np.random.choice(env.repair_action_space)
                else:
                    repair_action = repair_agent.take_action(state)
                repair_next_state, repair_reward, repair_done, _ = env.repair_step(repair_action, window_index=w, clean_data=clean_data, state=state, detected_errors=detected_errors)
                
              
                # 获取归一化的reward值
                normalized_direct = get_normalized_direct_clean_reward(direct_clean_reward)
                normalized_detection_repair = get_normalized_detection_repair_reward(detection_reward, repair_reward)
                
                print(f"  Strategy Comparison: raw_direct={direct_clean_reward:.4f}, norm_direct={normalized_direct:.4f}")
                print(f"  raw_det_repair={detection_reward + repair_reward:.4f}, norm_det_repair={normalized_detection_repair:.4f}")
                
                # 随机性以促进探索（10%的概率随机选择策略）
                if random.random() < 0.1:  # 10%的概率随机选择策略
                    strategy_choice = random.choice(["direct", "detection_repair"])
                    print(f"  Random exploration: choosing {strategy_choice}")
                else:
                    # 基于归一化后的值选择策略
                    if normalized_direct > normalized_detection_repair:
                        strategy_choice = "direct"
                    else:
                        strategy_choice = "detection_repair"
                    print(f"  Normalized comparison: choosing {strategy_choice}")
                
                # 更新reward历史记录
                update_reward_histories(direct_clean_reward, detection_reward, repair_reward)
                # ======================================================
                
                if strategy_choice == "direct":
                    # 选择直接清洗策略
                    direct_clean_transition_dict['direct_clean_states'].append(state)
                    direct_clean_transition_dict['direct_clean_actions'].append(direct_clean_action)
                    direct_clean_transition_dict['direct_clean_next_states'].append(direct_clean_next_state)
                    direct_clean_transition_dict['direct_clean_rewards'].append(direct_clean_reward)
                    direct_clean_transition_dict['direct_clean_dones'].append(done)
                    direct_clean_episode_return += direct_clean_reward
                    state = direct_clean_next_state
                else:
                    # 选择检测+修复策略
                    detection_transition_dict['detection_states'].append(state)
                    detection_transition_dict['detection_actions'].append(detection_action)
                    detection_transition_dict['detection_next_states'].append(state)
                    detection_transition_dict['detection_rewards'].append(detection_reward)
                    detection_transition_dict['detection_dones'].append(done)
                    detection_episode_return += detection_reward

                    repair_transition_dict['repair_states'].append(state)
                    repair_transition_dict['repair_actions'].append(repair_action)
                    repair_transition_dict['repair_next_states'].append(repair_next_state)
                    repair_transition_dict['repair_rewards'].append(repair_reward)
                    repair_transition_dict['repair_dones'].append(done)
                    repair_episode_return += repair_reward
                    state = repair_next_state

                step_counter += 1
            
            # 保存每个回合的return
            detection_return_list.append(detection_episode_return)
            repair_return_list.append(repair_episode_return)
            direct_clean_return_list.append(direct_clean_episode_return)

            # 向每个episode的总奖励中加入每个窗口的奖励
            total_detection_return += detection_episode_return
            total_repair_return += repair_episode_return
            total_direct_clean_return += direct_clean_episode_return
                
            # 更新策略网络
            if len(detection_transition_dict['detection_states']) > 0:
                detection_actor_loss, detection_critic_loss = detection_agent.update(detection_transition_dict=detection_transition_dict)
            else:
                detection_actor_loss, detection_critic_loss = 0.0, 0.0
                
            if len(repair_transition_dict['repair_states']) > 0:
                repair_actor_loss, repair_critic_loss = repair_agent.update(repair_transition_dict=repair_transition_dict)
            else:
                repair_actor_loss, repair_critic_loss = 0.0, 0.0
                
            if len(direct_clean_transition_dict['direct_clean_states']) > 0:
                direct_clean_actor_loss, direct_clean_critic_loss = direct_clean_agent.update(direct_clean_transition_dict=direct_clean_transition_dict)
            else:
                direct_clean_actor_loss, direct_clean_critic_loss = 0.0, 0.0

            w += random.randint(1, 2) + 1
    
        return (total_detection_return, total_repair_return, total_direct_clean_return, 
                detection_actor_loss, detection_critic_loss, 
                repair_actor_loss, repair_critic_loss,
                direct_clean_actor_loss, direct_clean_critic_loss)


def test():
    """测试过程，包含对所有训练模型的评估"""
    # 重置reward历史记录（测试时重新开始记录）
    global global_direct_clean_reward_history, global_detection_repair_reward_history
    global_direct_clean_reward_history = deque(maxlen=200)
    global_detection_repair_reward_history = deque(maxlen=200)
    
    # 加载测试数据集
    test_dirty_data_df = pd.read_csv(test_dirty_dataset, header=None, low_memory=False)
    test_clean_data_df = pd.read_csv(test_clean_dataset, header=None, low_memory=False)

    # 获取列名（如果有）
    column_names = test_clean_data_df.iloc[0, 1:].values if test_clean_data_df.shape[1] > 1 else None
    
    # 提取数据部分（跳过第一行表头）
    test_dirty_data = test_dirty_data_df.iloc[1:, 1:].values.astype(np.float64)
    test_clean_data = test_clean_data_df.iloc[1:, 1:].values.astype(np.float64)
    
    # 获取实际的列数
    num_test_columns = test_clean_data.shape[1]
    
    # 初始化scaler并归一化数据
    test_scaler = MinMaxScaler()
    test_clean_data_normalized = test_scaler.fit_transform(test_clean_data)
    test_dirty_data_normalized = test_scaler.transform(test_dirty_data)
    
    # 创建测试环境（使用实际的列数）
    test_env = DataCleaningEnv(test_clean_data_normalized, window_size=window_sizes)

    # 保存评估结果到 CSV
    eval_results_path = "all_models_evaluation_results.csv"
    eval_header = ["Model Epoch", "Test Dataset", "MSE", "MNAD", "RRA", 
                  "Precision", "Recall", "F1", "MAE Error", "Time Cost"]

    if not os.path.exists(eval_results_path):
        with open(eval_results_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(eval_header)
    
    direct_model_para_dir = "comprehensive_model_para_ene_real_fur_meta"
    dual_model_para_dir = "comprehensive_model_para_ene_real_fur_meta"
    
    # 获取所有模型文件并按epoch排序
    direct_model_files = natsorted([f for f in os.listdir(direct_model_para_dir) 
                              if f.startswith("direct_clean_actor_parameters") and f.endswith(".pth")])
    detection_model_files = natsorted([f for f in os.listdir(dual_model_para_dir) 
                                 if f.startswith("detection_actor_parameters") and f.endswith(".pth")])
    repair_model_files = natsorted([f for f in os.listdir(dual_model_para_dir) 
                              if f.startswith("repair_actor_parameters") and f.endswith(".pth")])

    # 确保三个模型的轮次数量一致
    num_epochs = min(len(direct_model_files), len(detection_model_files), len(repair_model_files))
    print(f"Found {num_epochs} saved models to evaluate")

    # 对每个保存的模型进行评估
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n=== Evaluating model from epoch {epoch + 1}/{num_epochs} ===")
        
        # 加载当前epoch的模型
        current_direct_model = direct_model_files[epoch]
        current_detection_model = detection_model_files[epoch]
        current_repair_model = repair_model_files[epoch]
        
        # 创建元模型
        meta_detection_agent, meta_repair_agent, meta_direct_clean_agent = create_agent(test_env, num_test_columns)
        
        # 加载模型参数
        detection_state_dict = torch.load(os.path.join(dual_model_para_dir, current_detection_model))
        meta_detection_agent = load_compatible_weights(meta_detection_agent, detection_state_dict)
        
        repair_state_dict = torch.load(os.path.join(dual_model_para_dir, current_repair_model))
        meta_repair_agent = load_compatible_weights(meta_repair_agent, repair_state_dict)
        
        direct_clean_state_dict = torch.load(os.path.join(direct_model_para_dir, current_direct_model))
        meta_direct_clean_agent = load_compatible_weights(meta_direct_clean_agent, direct_clean_state_dict)
        
        # 创建任务特定模型
        task_detection_agent = ActorCritic(
            num_column=num_test_columns,
            window_size=window_sizes,
            n_hiddens=n_hiddens,
            n_actions=test_env.detection_action_space,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            device=device
        ).to(device)
        
        task_repair_agent = ActorCritic(
            num_column=num_test_columns,
            window_size=window_sizes,
            n_hiddens=n_hiddens,
            n_actions=test_env.repair_action_space,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            device=device
        ).to(device)
        
        task_direct_clean_agent = ActorCritic(
            num_column=num_test_columns,
            window_size=window_sizes,
            n_hiddens=n_hiddens,
            n_actions=test_env.action_space,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            device=device
        ).to(device)
        
        # 初始化任务模型参数
        task_detection_agent.load_state_dict(meta_detection_agent.state_dict())
        task_repair_agent.load_state_dict(meta_repair_agent.state_dict())
        task_direct_clean_agent.load_state_dict(meta_direct_clean_agent.state_dict())
        
        # 对新任务进行少量梯度更新（元学习适配）
        print("Performing meta-learning adaptation...")
        for adapt_step in range(5):
            # 随机选择一些窗口进行适应
            num_windows = int(test_clean_data_normalized.shape[0] / window_sizes)
            adapt_windows = random.sample(range(num_windows), min(5, num_windows))
            
            # 在这些窗口上收集经验
            detection_transitions = {
                'detection_states': [], 'detection_actions': [], 
                'detection_next_states': [], 'detection_rewards': [], 
                'detection_dones': []
            }
            repair_transitions = {
                'repair_states': [], 'repair_actions': [], 
                'repair_next_states': [], 'repair_rewards': [], 
                'repair_dones': []
            }
            direct_clean_transitions = {
                'direct_clean_states': [], 'direct_clean_actions': [], 
                'direct_clean_next_states': [], 'direct_clean_rewards': [], 
                'direct_clean_dones': []
            }
            
            for w in adapt_windows:
                state, _ = test_env.reset(window_index=w)
                done = False
                step_counter = 0
                
                while not done and step_counter <= 5:
                    # 直接清洗策略
                    direct_action = task_direct_clean_agent.take_action(state)
                    direct_next, direct_reward, direct_done, _ = test_env.step(
                        direct_action, w, test_clean_data_normalized, state)
                    
                    # 检测+修复策略
                    det_action = task_detection_agent.take_action(state)
                    errors, det_reward = test_env.detection_step(
                        det_action, w, test_clean_data_normalized, state)
                    repair_action = task_repair_agent.take_action(state)
                    repair_next, repair_reward, repair_done, _ = test_env.repair_step(
                        repair_action, w, test_clean_data_normalized, state, errors)
                    
                    
                    # 获取归一化的reward值
                    norm_direct = get_normalized_direct_clean_reward(direct_reward)
                    norm_det_repair = get_normalized_detection_repair_reward(det_reward, repair_reward)
                    
                    # 更新历史记录
                    update_reward_histories(direct_reward, det_reward, repair_reward)
                    
                    if norm_direct >= norm_det_repair:
                        # 使用直接清洗
                        direct_clean_transitions['direct_clean_states'].append(state)
                        direct_clean_transitions['direct_clean_actions'].append(direct_action)
                        direct_clean_transitions['direct_clean_next_states'].append(direct_next)
                        direct_clean_transitions['direct_clean_rewards'].append(direct_reward)
                        direct_clean_transitions['direct_clean_dones'].append(done)
                        state = direct_next
                        done = direct_done
                    else:
                        # 使用检测+修复
                        detection_transitions['detection_states'].append(state)
                        detection_transitions['detection_actions'].append(det_action)
                        detection_transitions['detection_next_states'].append(state)
                        detection_transitions['detection_rewards'].append(det_reward)
                        detection_transitions['detection_dones'].append(done)

                        repair_transitions['repair_states'].append(state)
                        repair_transitions['repair_actions'].append(repair_action)
                        repair_transitions['repair_next_states'].append(repair_next)
                        repair_transitions['repair_rewards'].append(repair_reward)
                        repair_transitions['repair_dones'].append(done)
                        state = repair_next
                        done = repair_done
                    # ======================================================
                    
                    step_counter += 1
            
            # 更新任务模型
            if len(detection_transitions['detection_states']) > 0:
                task_detection_agent.update(detection_transition_dict=detection_transitions)
            if len(repair_transitions['repair_states']) > 0:
                task_repair_agent.update(repair_transition_dict=repair_transitions)
            if len(direct_clean_transitions['direct_clean_states']) > 0:
                task_direct_clean_agent.update(direct_clean_transition_dict=direct_clean_transitions)
        
        # 使用当前模型进行数据清洗
        clean_start_time = time.time()
        cleaned_data_all = []
        num_windows = int(test_clean_data_normalized.shape[0] / window_sizes)
        
        # 重置reward历史用于测试阶段的比较
        global_direct_clean_reward_history = deque(maxlen=200)
        global_detection_repair_reward_history = deque(maxlen=200)
        
        for w in range(num_windows):
            test_env.direct_dirty_data = test_dirty_data_normalized
            state, _ = test_env.reset_with_only_window(window_index=w)
            done = False
            step_counter = 0
            
            while not done and step_counter <= 5:
                # 获取两种策略的动作和奖励
                direct_action = task_direct_clean_agent.take_action(state)
                direct_next, direct_reward, direct_done, _ = test_env.step(
                    direct_action, w, test_clean_data_normalized, state)
                
                det_action = task_detection_agent.take_action(state)
                errors, det_reward = test_env.detection_step(
                    det_action, w, test_clean_data_normalized, state)
                repair_action = task_repair_agent.take_action(state)
                repair_next, repair_reward, repair_done, _ = test_env.repair_step(
                    repair_action, w, test_clean_data_normalized, state, errors)
                
                
                norm_direct = get_normalized_direct_clean_reward(direct_reward)
                norm_det_repair = get_normalized_detection_repair_reward(det_reward, repair_reward)
                
                # 更新历史记录
                update_reward_histories(direct_reward, det_reward, repair_reward)
                
                if norm_direct >= norm_det_repair:
                    state = direct_next
                    done = direct_done
                else:
                    state = repair_next
                    done = repair_done
                # ======================================================
                
                step_counter += 1
            
            cleaned_data_all.append(state)
        
        # 处理清洗结果
        cleaned_data_all = np.vstack(cleaned_data_all)
        clean_time = time.time() - clean_start_time
        
        # 恢复归一化
        cleaned_data_original = test_scaler.inverse_transform(cleaned_data_all)
        dirty_data_original = test_scaler.inverse_transform(test_dirty_data_normalized)
        
        # 保存当前模型的清洗结果
        os.makedirs("all_models_cleaned_results", exist_ok=True)
        dataset_name = os.path.basename(test_dirty_dataset).replace('.csv', '')
        save_path = os.path.join("all_models_cleaned_results", 
                                f"cleaned_epoch{epoch+1}_{dataset_name}.csv")
        
        if column_names is not None:
            pd.DataFrame(cleaned_data_original, columns=column_names).to_csv(save_path, index=False)
        else:
            pd.DataFrame(cleaned_data_original).to_csv(save_path, index=False)
        
        # 计算评估指标
        clean_data_original = test_scaler.inverse_transform(test_clean_data_normalized)
        mse_val = mse(cleaned_data_original, clean_data_original)
        mnad_val = mnad(cleaned_data_original, clean_data_original)
        rra_val = rra(cleaned_data_original, clean_data_original, dirty_data_original)
        prec, rec, f1_val = precision_recall(cleaned_data_original, clean_data_original, dirty_data_original)
        mae_val = mae_error(cleaned_data_original, clean_data_original, dirty_data_original)
        epoch_time = time.time() - epoch_start_time
        
        # 打印评估结果
        print("\nEvaluation Metrics:")
        print(f"MSE: {mse_val:.4f} | MNAD: {mnad_val:.4f} | RRA: {rra_val:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1_val:.4f}")
        print(f"MAE: {mae_val:.4f} | Time: {epoch_time:.2f}s")
        
        # 保存评估结果
        with open(eval_results_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch + 1,  # 模型训练轮次
                dataset_name,
                mse_val, 
                mnad_val, 
                rra_val,
                prec, 
                rec, 
                f1_val,
                mae_val,
                epoch_time
            ])
        
        print(f"Epoch {epoch + 1} evaluation saved to {save_path}")

    print("\nAll models evaluation completed!")

if __name__ == '__main__':
    #meta_train()
    test()
