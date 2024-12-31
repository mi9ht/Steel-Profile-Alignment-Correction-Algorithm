from collections import deque
import numpy as np
import random


# 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 队列，淘汰古老数据

    def __len__(self):
        return len(self.buffer)

    # 将数据加入经验池
    def add(self, state, action1, action2, reward1, reward2, next_state, done):
        self.buffer.append((state, action1, action2, reward1, reward2, next_state, done))

    # 采样，大小为 batch_size
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state,  action1, action2, reward1, reward2, next_state, done = zip(*batch)
        return np.array(state), action1, action2, reward1, reward2, np.array(next_state), done


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
