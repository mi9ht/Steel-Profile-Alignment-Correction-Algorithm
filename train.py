from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from net import *
from environment import *
from utils import *


# 设置超参数
lr = 1e-5
num_episodes = 200000
state_dim = 2
hidden_dim = 1024
action_dim1 = 46
action_dim2 = 21
num_layers = 2
gamma = 0.1
epsilon = 0.01
target_update = 100
buffer_size = 50000
minimal_size = 5000
batch_size = 512
model = 'Double DQN'


path = 'net_{}_{}_{}'.format(model, k, mu)


# 训练
def train(agent, env):
    replay_buffer = ReplayBuffer(buffer_size)
    return_list1, return_list2 = [], []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return1, episode_return2 = 0, 0
                state = env.reset()
                done = False
                while not done:
                    action1, action2 = agent.take_action(state, num_episodes / 10 * i + i_episode + 1)
                    next_state, reward1, reward2, done, _ = env.step(state, action1, action2)
                    replay_buffer.add(state, action1, action2, reward1, reward2, next_state, done)
                    state = next_state
                    episode_return1 += reward1
                    episode_return2 += reward2
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if len(replay_buffer) > minimal_size:
                        b_s, b_a1, b_a2, b_r1, b_r2, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions1': b_a1,
                            'actions2': b_a2,
                            'next_states': b_ns,
                            'rewards1': b_r1,
                            'rewards2': b_r2,
                            'dones': b_d
                        }  # 批量采样
                        agent.update(transition_dict)  # 批量更新
                return_list1.append(episode_return1)
                return_list2.append(episode_return2)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return1':
                        '%.2f' % np.mean(return_list1[-10:]),
                        'return2':
                        '%.2f' % np.mean(return_list2[-10:]),
                        'ops':
                        '%.d' % env.get_n_ops()
                    })
                pbar.update(1)
    return [a + b for a, b in zip(return_list1, return_list2)]
    # return return_list1


def test(agent, env):
    test_nums = 10000
    cnt1 = 0
    cnt2 = 0
    for _ in range(test_nums):
        state = env.reset()
        action1, actino2 = agent.best_action(state)
        next_state, _, _, _, _ = env.step(state, action1, actino2)
        v_s = next_state[1] * (1 - k * (1 - mu))
        sin_v = (abs(next_state[0]) * v_s) / steel_length
        if sin_v > 1:
            break
        angle = np.degrees(np.arcsin(sin_v))
        if next_state[1] != -1 and angle <= 1:
            cnt1 += 1
        if next_state[1] != -1 and angle <= 3:
            cnt2 += 1
    print('1-成功率: {:.2f}%'.format(cnt1 / test_nums * 100))
    print('3-成功率: {:.2f}%'.format(cnt2 / test_nums * 100))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

if __name__ == '__main__':
    set_seed(0)
    
    env = MyEnv(action_dim1, action_dim2)
    agent = Agent(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim1=action_dim1,
        action_dim2=action_dim2,
        lr=lr,
        gamma=gamma,
        e=epsilon,
        target_update=target_update,
        model=model,
        num_layers=num_layers
    )

    agent.load_net('论文/实验/0.2 0.6/net_Double DQN_0.2_0.6_10')
    agent.q_net1.train()
    agent.q_net2.train()
    # agent.q_net.train()
    return_list = train(agent, env)
    return_list_pd = pd.DataFrame(return_list, columns=[None])
    return_list_pd.to_excel('reward_{}_{}_{}.xlsx'.format(model, MAX_STEP, mu), header=False, index=False)
    
    mv_return = moving_average(return_list, num_episodes // 100 - 1)
    plt.plot(range(len(mv_return)), mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()
    
    agent.save_net(path)

    for _ in range(1):
        test(agent, env)
