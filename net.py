import torch
import torch.nn as nn
import numpy as np


# 设备参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Q网络
class Q_Net(torch.nn.Module):
    def __init__(self, state_dim, num_layers, hidden_dim, action_dim):
        super().__init__()
        # 共享网络层
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.fc_fin = torch.nn.Linear(hidden_dim, action_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_fin(x)
        return x

class Q_Net2(torch.nn.Module):
    def __init__(self, state_dim, num_layers, hidden_dim, action_dim1, action_dim2):
        super().__init__()
        # 共享网络层
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.fc_fin1 = torch.nn.Linear(hidden_dim, action_dim1)
        self.fc_fin2 = torch.nn.Linear(hidden_dim, action_dim2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x1 = self.fc_fin1(x)
        x2 = self.fc_fin2(x)
        return x1, x2


# VA网络
class VA_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_layers = 3):
        super().__init__()
        # 共享网络层
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        
        self.fc_V = torch.nn.Linear(hidden_dim, 1)  # V 网络
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)  # A 网络
        self.relu = torch.nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.fc(x)
        V = self.fc_V(x)
        A = self.fc_A(x)
        Q = V + A - A.mean()
        return Q


class Agent:
    def __init__(self, state_dim, hidden_dim, action_dim1, action_dim2, lr, gamma, e, target_update, model, num_layers=3):
        self.lr = lr
        self.gamma = gamma
        self.e = e
        self.target_update = target_update
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        self.model = model
        self.count = 0

        if model == 'DQN' or model == 'Double DQN':
            # Q网络
            self.q_net1 = Q_Net(state_dim, num_layers, hidden_dim, action_dim1).to(device)
            self.q_net2 = Q_Net(state_dim, num_layers, hidden_dim, action_dim2).to(device)
            # target网络
            self.target_q_net1 = Q_Net(state_dim, num_layers, hidden_dim, action_dim1).to(device)
            self.target_q_net2 = Q_Net(state_dim, num_layers, hidden_dim, action_dim2).to(device)
        else:
            # VA网络
            self.q_net1 = VA_Net(state_dim, hidden_dim, action_dim1, num_layers).to(device)
            self.q_net2 = VA_Net(state_dim, hidden_dim, action_dim2, num_layers).to(device)
            # target网络
            self.target_q_net1 = VA_Net(state_dim, hidden_dim, action_dim1, num_layers).to(device)
            self.target_q_net2 = VA_Net(state_dim, hidden_dim, action_dim2, num_layers).to(device)

        # 优化器
        self.optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=self.lr)

        # 损失函数
        self.loss_fc = torch.nn.MSELoss()

    # e-greedy贪婪策略选取动作
    def take_action(self, state, episode):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        epsilon = max(self.e * np.exp(-0.0002 * episode), 0.01)
        # epsilon = 0.01
        if np.random.random() < epsilon:
            action1 = np.random.randint(self.action_dim1)
        else:
            action1 = self.q_net1(state).argmax().item()
        if np.random.random() < epsilon:
            action2 = np.random.randint(self.action_dim2)
        else:
            action2 = self.q_net2(state).argmax().item()
        return action1, action2


    # 批量更新
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(device)
        actions1 = torch.tensor(transition_dict['actions1']).view(-1, 1).to(
            device)
        actions2 = torch.tensor(transition_dict['actions2']).view(-1, 1).to(
            device)
        rewards1 = torch.tensor(transition_dict['rewards1'],
                               dtype=torch.float).view(-1, 1).to(device)
        rewards2 = torch.tensor(transition_dict['rewards2'],
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(device)

        # 训练的Q值
        q_values1 = self.q_net1(states).gather(dim=1, index=actions1)
        q_values2 = self.q_net2(states).gather(dim=1, index=actions2)
        # 目标Q值
        if self.model == 'DQN' or self.model == 'Dueling DQN':
            max_target_q_values1 = self.target_q_net1(next_states).max(dim=1)[0].view(-1, 1)
            max_target_q_values2 = self.target_q_net2(next_states).max(dim=1)[0].view(-1, 1)
        else:
            max_action1 = self.q_net1(next_states).max(1)[1].view(-1, 1)
            max_action2 = self.q_net2(next_states).max(1)[1].view(-1, 1)
            max_target_q_values1 = self.target_q_net1(next_states).gather(1, max_action1)
            max_target_q_values2 = self.target_q_net2(next_states).gather(1, max_action2)
        # TD-target
        targets1 = rewards1 + self.gamma * max_target_q_values1 * (1 - dones)
        targets2 = rewards2 + self.gamma * max_target_q_values2 * (1 - dones)
        # 计算损失
        loss = self.loss_fc(q_values1, targets1) + self.loss_fc(q_values2, targets2)
        # 梯度清零（pytorth默认梯度累积）
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        self.optimizer1.step()
        self.optimizer2.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net1.load_state_dict(self.q_net1.state_dict())
            self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        self.count = (self.count + 1) % self.target_update
    
    # 批量更新1
    def update1(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(device)
        actions1 = torch.tensor(transition_dict['actions1']).view(-1, 1).to(
            device)
        actions2 = torch.tensor(transition_dict['actions2']).view(-1, 1).to(
            device)
        rewards1 = torch.tensor(transition_dict['rewards1'],
                               dtype=torch.float).view(-1, 1).to(device)
        rewards2 = torch.tensor(transition_dict['rewards2'],
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(device)

        # 训练的Q值
        q_values1 = self.q_net1(states).gather(dim=1, index=actions1)
        # 目标Q值
        if self.model == 'DQN' or self.model == 'Dueling DQN':
            max_target_q_values1 = self.target_q_net1(next_states).max(dim=1)[0].view(-1, 1)
        else:
            max_action1 = self.q_net1(next_states).max(1)[1].view(-1, 1)
            max_target_q_values1 = self.target_q_net1(next_states).gather(1, max_action1)
        # TD-target
        targets1 = rewards1 + self.gamma * max_target_q_values1 * (1 - dones)
        # 计算损失
        loss1 = self.loss_fc(q_values1, targets1)
        # 梯度清零（pytorth默认梯度累积）
        self.optimizer1.zero_grad()
        # 反向传播
        loss1.backward()
        # 更新参数
        self.optimizer1.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.count = (self.count + 1) % self.target_update
        
    # 获取当前状态下最优策略
    def best_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        action1 = self.q_net1(state).argmax().item()
        action2 = self.q_net2(state).argmax().item()
        return action1, action2
    
    # 保存Q网络模型参数
    def save_net(self, path):
        torch.save(self.q_net1.state_dict(), path + '_1.pth')
        torch.save(self.q_net2.state_dict(), path + '_2.pth')
        
    # 加载Q网络模型参数
    def load_net(self, path):
        self.q_net1.load_state_dict(torch.load(path + '_1.pth'))
        self.q_net1.eval()
        self.q_net2.load_state_dict(torch.load(path + '_2.pth'))
        self.q_net2.eval()
        
class Agent2:
    def __init__(self, state_dim, hidden_dim, action_dim1, action_dim2, lr, gamma, e, target_update, model, num_layers=3):
        self.lr = lr
        self.gamma = gamma
        self.e = e
        self.target_update = target_update
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        self.model = model
        self.count = 0

        if model == 'DQN' or model == 'Double DQN':
            # Q网络
            self.q_net = Q_Net2(state_dim, num_layers, hidden_dim, action_dim1, action_dim2).to(device)
            # target网络
            self.target_q_net = Q_Net2(state_dim, num_layers, hidden_dim, action_dim1, action_dim2).to(device)


        # 优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        # 损失函数
        self.loss_fc = torch.nn.MSELoss()

    # e-greedy贪婪策略选取动作
    def take_action(self, state, episode):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        epsilon = max(self.e * np.exp(-0.0005 * episode), 0.1)
        # epsilon = 0.01
        q1, q2 = self.q_net(state)
        if np.random.random() < epsilon:
            action1 = np.random.randint(self.action_dim1)
        else:
            action1 = q1.argmax().item()
        if np.random.random() < epsilon:
            action2 = np.random.randint(self.action_dim2)
        else:
            action2 = q2.argmax().item()
        return action1, action2


    # 批量更新
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(device)
        actions1 = torch.tensor(transition_dict['actions1']).view(-1, 1).to(
            device)
        actions2 = torch.tensor(transition_dict['actions2']).view(-1, 1).to(
            device)
        rewards1 = torch.tensor(transition_dict['rewards1'],
                               dtype=torch.float).view(-1, 1).to(device)
        rewards2 = torch.tensor(transition_dict['rewards2'],
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(device)

        # 训练的Q值
        q1, q2 = self.q_net(states)
        q_values1 = q1.gather(dim=1, index=actions1)
        q_values2 = q2.gather(dim=1, index=actions2)
        # 目标Q值
        if self.model == 'DQN' or self.model == 'Dueling DQN':
            max_target_q1, max_target_q2 = self.target_q_net(next_states)
            max_target_q_values1 = max_target_q1.max(dim=1)[0].view(-1, 1)
            max_target_q_values2 = max_target_q2.max(dim=1)[0].view(-1, 1)
        else:
            nq1, nq2 = self.q_net(next_states)
            max_action1 = nq1.max(1)[1].view(-1, 1)
            max_action2 = nq2.max(1)[1].view(-1, 1)
            max_target_q1, max_target_q2 = self.target_q_net(next_states)
            max_target_q_values1 = max_target_q1.gather(1, max_action1)
            max_target_q_values2 = max_target_q2.gather(1, max_action2)
        # TD-target
        targets1 = rewards1 + self.gamma * max_target_q_values1 * (1 - dones)
        targets2 = rewards2 + self.gamma * max_target_q_values2 * (1 - dones)
        # 计算损失
        loss = self.loss_fc(q_values1, targets1) + self.loss_fc(q_values2, targets2)
        # 反向传播
        loss.backward()
        # 更新参数
        self.optimizer.step()
        # 梯度清零（pytorth默认梯度累积）
        self.optimizer.zero_grad()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count = (self.count + 1) % self.target_update
        
    # 获取当前状态下最优策略
    def best_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        q1, q2 = self.q_net(state)
        action1 = q1.argmax().item()
        action2 = q2.argmax().item()
        return action1, action2
    
    # 保存Q网络模型参数
    def save_net(self, path):
        torch.save(self.q_net.state_dict(), path + '.pth')
        
    # 加载Q网络模型参数
    def load_net(self, path):
        self.q_net.load_state_dict(torch.load(path + '.pth'))
        self.q_net.eval()
        