import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from train import *

# df1 = pd.read_excel('论文/实验/0.2 0.2/reward_Double DQN_10_0.2.xlsx', header=None)
# df2 = pd.read_excel('论文/实验/0.2 0.4/reward_Double DQN_10_0.4.xlsx', header=None)
# df3 = pd.read_excel('论文/实验/0.2 0.6/reward_Double DQN_10_0.6.xlsx', header=None)

# return_list1 = df1[0].tolist()
# return_list2 = df2[0].tolist()
# return_list3 = df3[0].tolist()

# mv_return1 = moving_average(return_list1, len(return_list1) // 100 - 1)
# mv_return2 = moving_average(return_list2, len(return_list2) // 100 - 1)
# mv_return3 = moving_average(return_list3, len(return_list3) // 100 - 1)
# plt.plot(range(len(mv_return2)), mv_return2, label='μ=0.2')
# plt.plot(range(len(mv_return1)), mv_return1, label='μ=0.4')
# plt.plot(range(len(mv_return3)), mv_return3, label='μ=0.6')

env = MyEnv(action_dim1, action_dim2)
agent1 = Agent(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim1=action_dim1,
        action_dim2=action_dim2,
        lr=lr,
        gamma=gamma,
        e=epsilon,
        target_update=target_update,
        model='Double DQN',
        num_layers=num_layers
    )
agent1.load_net('论文/实验/0.2 0.2/net_Double DQN_0.2_0.2')

agent2 = Agent(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim1=action_dim1,
        action_dim2=action_dim2,
        lr=lr,
        gamma=gamma,
        e=epsilon,
        target_update=target_update,
        model='DQN',
        num_layers=num_layers
    )
agent2.load_net('论文/实验/DQN 0.2/net_DQN_0.2_0.2')

acc1, acc2 = [], []

def test(agent, eps):
    cnt = 0
    for _ in range(10000):
        state = env.reset()
        action1, actino2 = agent.best_action(state)
        next_state, _, _, _, _ = env.step(state, action1, actino2)
        v_s = next_state[1] * (1 - k * (1 - mu))
        sin_v = (abs(next_state[0]) * v_s) / steel_length
        if sin_v > 1:
            break
        angle = np.degrees(np.arcsin(sin_v))
        if next_state[1] != -1 and angle <= eps:
            cnt += 1
    return cnt / 10000

for i in range(1, 6):
    print(test(agent1, i), test(agent2, i))
    acc1.append(test(agent1, i))
    acc2.append(test(agent2, i))

plt.plot(range(1, 6), acc2, label='DQN')
plt.plot(range(1, 6), acc1, label='DDQN')
plt.grid(which='both', axis='both')
plt.xticks(range(1, 6))
plt.xlabel('threshold value')
plt.ylabel('success rate')

plt.legend()
plt.show()