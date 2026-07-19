import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tcn import TemporalConvNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# ------------------------------------
# 策略梯度Actor，动作选择
# ------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, num_column, window_size, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        assert num_column > 0, "num_column 必须大于 0"
        self.tcn = TemporalConvNet(num_inputs=num_column,
                                   num_channels=[num_column, 64, 10, 1],
                                   kernel_size=3,
                                   dropout=0.3)
        self.fc1 = nn.Linear(window_size, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        # x shape: [batch_size, num_column, window_size]
        x = self.tcn(x).squeeze(1)  # [batch_size, hidden_dim]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# ------------------------------------
# 值函数Critic，动作评估输出 shape=[b,1]
# ------------------------------------
class ValueNet(nn.Module):
    def __init__(self, num_column, window_size, n_hiddens):
        super(ValueNet, self).__init__()
        assert num_column > 0, "num_column 必须大于 0"
        self.tcn = TemporalConvNet(num_inputs=num_column,
                                   num_channels=[num_column, 64, 10, 1],
                                   kernel_size=3,
                                   dropout=0.3)
        self.fc1 = nn.Linear(window_size, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.tcn(x).squeeze(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# ------------------------------------
# Actor-Critic 主体
# ------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, num_column, window_size, n_hiddens, n_actions,
                 actor_lr, critic_lr, gamma, device):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.num_column = num_column
        self.window_size = window_size
        self.n_hiddens = n_hiddens
        self.n_actions = n_actions
        self.device = device

        # 实例化策略网络
        self.actor = PolicyNet(num_column=num_column,
                              window_size=window_size,
                              n_hiddens=n_hiddens,
                              n_actions=n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(num_column=num_column,
                              window_size=window_size,
                              n_hiddens=n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state):
        assert len(state.shape) == 2, f"State must be 2D, got {state.shape}"
        assert state.shape[1] == self.num_column, f"Expected {self.num_column} columns, got {state.shape[1]}"
        state = torch.tensor(state[np.newaxis, :, :], dtype=torch.float).to(self.device)  # [1, window_size, num_column]
        state = state.transpose(1, 2)  # [1, num_column, window_size]
        with torch.no_grad():
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
        return action

    def update(self, detection_transition_dict=None, repair_transition_dict=None, direct_clean_transition_dict=None):
        if detection_transition_dict is not None:
            states = np.array(detection_transition_dict['detection_states'])
            actions = detection_transition_dict['detection_actions']
            rewards = detection_transition_dict['detection_rewards']
            next_states = np.array(detection_transition_dict['detection_next_states'])
            dones = detection_transition_dict['detection_dones']
        elif repair_transition_dict is not None:
            states = np.array(repair_transition_dict['repair_states'])
            actions = repair_transition_dict['repair_actions']
            rewards = repair_transition_dict['repair_rewards']
            next_states = np.array(repair_transition_dict['repair_next_states'])
            dones = repair_transition_dict['repair_dones']
        elif direct_clean_transition_dict is not None:
            states = np.array(direct_clean_transition_dict['direct_clean_states'])
            actions = direct_clean_transition_dict['direct_clean_actions']
            rewards = direct_clean_transition_dict['direct_clean_rewards']
            next_states = np.array(direct_clean_transition_dict['direct_clean_next_states'])
            dones = direct_clean_transition_dict['direct_clean_dones']
        else:
            return 0.0, 0.0  # 默认返回

        print("States shape before processing:", states.shape)

        # 检查是否为空
        if len(states) == 0:
            print("Warning: Empty transition dict received.")
            return 0.0, 0.0

        # 确保 states 是 3D 张量
        if len(states.shape) == 1:
            # 如果是 [feature_dim]，扩展为 [1, feature_dim, 1]
            states = np.expand_dims(np.expand_dims(states, axis=0), axis=2)
            next_states = np.expand_dims(np.expand_dims(next_states, axis=0), axis=2)
            print("Expanded from 1D to [1, feature_dim, 1]")
        elif len(states.shape) == 2:
            # 如果是 [batch_size, feature_dim]，扩展为 [batch_size, feature_dim, 1]
            states = np.expand_dims(states, axis=2)
            next_states = np.expand_dims(next_states, axis=2)
            print("Expanded states to 3D")
        elif len(states.shape) == 4:
            # 如果是 [batch_size, 1, feature_dim, window_size]，压缩到 [batch_size, feature_dim, window_size]
            states = states.squeeze(1)
            next_states = next_states.squeeze(1)
            print("Squeezed extra dimension")

        # 确保通道顺序是 [batch_size, num_column, window_size]
        if states.shape[1] != self.num_column and states.shape[2] == self.num_column:
            states = np.transpose(states, (0, 2, 1))
            next_states = np.transpose(next_states, (0, 2, 1))
            print("Permuted states from [B, W, C] to [B, C, W]")

        # 再次检查
        assert len(states.shape) == 3, f"Expected 3D input after processing, got {states.shape}"
        assert states.shape[1] == self.num_column, \
            f"Channel mismatch after processing: {states.shape[1]} != {self.num_column}"

        # 转换为 PyTorch Tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        # 计算 TD 目标与误差
        td_value = self.critic(states)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - td_value

        # 策略梯度损失
        log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8)
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # 值函数损失
        critic_loss = torch.mean(F.mse_loss(td_value, td_target.detach()))

        # 更新参数
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())


if __name__ == '__main__':
    # 测试模型构建和运行
    num_column = 38
    window_size = 20
    n_hiddens = 64
    n_actions = 3
    actor_lr = 1e-2
    critic_lr = 1e-3
    gamma = 0.9

    policy_net = PolicyNet(num_column, window_size, n_hiddens, n_actions)
    value_net = ValueNet(num_column, window_size, n_hiddens)
    actor_critic = ActorCritic(num_column, window_size, n_hiddens, n_actions, actor_lr, critic_lr, gamma)

    # 测试输入
    test_state = np.random.randn(window_size, num_column)
    action = actor_critic.take_action(test_state)
    print("Action taken:", action)

    # 构造 transition dict
    dummy_transitions = {
        'direct_clean_states': [test_state],
        'direct_clean_actions': [action],
        'direct_clean_rewards': [1.0],
        'direct_clean_next_states': [test_state],
        'direct_clean_dones': [False]
    }

    actor_loss, critic_loss = actor_critic.update(direct_clean_transition_dict=dummy_transitions)
    print("Actor Loss:", actor_loss)
    print("Critic Loss:", critic_loss)

