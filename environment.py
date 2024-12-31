import random
import numpy as np


# 环境参数
L = 4  # 4m
steel_length = 2  # 2m
t_machine = 2  # 2s
mu = 0.2
k = 0.2
MAX_STEP = 1
a1_min, a1_max = 15, 50  # 15 ~ 50s
a2_min, a2_max = -10, 10  # -10 ~ 10s


# 自定义环境
class MyEnv:
    def __init__(self, action_dim1, action_dim2):
        self.cnt = 0
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        
    # 离散转连续
    def dis_to_con(self, discrete_time, action_dim, time_min, time_max):
        return time_min + (discrete_time / (action_dim - 1)) * (time_max - time_min)
    
    def reset(self):
        self.n_ops = 0
        v = random.randint(10, 20) / 100  # 0.1m/s ~ 0.2m/s
        v_s = v * (1 - k * (1 - mu))
        angle = random.uniform(-30, 30)
        radians = np.radians(angle)
        delta_t = (np.sin(radians) * steel_length) / v_s
        # print(delta_t, v, v_s, angle)
        return np.array([delta_t, v], dtype=float)
    
    def step(self, state, action1, action2):
        delta_t, v = state[0], state[1]
        v_s = v * (1 - k * (1 - mu))
        t_transit = L / v_s
        a1 = self.dis_to_con(action1, self.action_dim1, a1_min, a1_max)
        if a1 + t_machine > t_transit:
            return np.array([delta_t, -1], dtype=float), -100, -100, True, t_transit
        t_gap = t_transit - a1 - t_machine
        reward1 = -t_gap * 4
        a2 = self.dis_to_con(action2, self.action_dim2, a2_min, a2_max)
        if a2 > 0:
            new_delta_t = delta_t - (a2 - min(a2, t_gap))
        else:
            new_delta_t = delta_t - (a2 + min(-a2, t_gap))
        next_state = np.array([new_delta_t, v], dtype=float)
        reward2 = -abs(new_delta_t) * 5 - self.n_ops * 5
        self.n_ops += 1
        # print(delta_t, new_delta_t, t_transit, a1, a2, reward1, reward2, t_gap)
        sin_v = ((abs(new_delta_t) * v_s) / steel_length)
        if sin_v > 1  or np.degrees(np.arcsin(sin_v)) > 30:
            return next_state, reward1, -100, True, t_transit
        done = (t_gap < 1 and np.degrees(np.arcsin(sin_v)) < 1) or self.n_ops >= MAX_STEP
        return next_state, reward1, reward2, done, t_transit
    
    def reset_n_ops(self):
        self.n_ops = 0
    
    def get_n_ops(self):
        return self.n_ops


if __name__ == '__main__':
    env = MyEnv(46, 21)
    # for i in range(10):
    #     env.reset()
    
    state = env.reset()
    a, b = 0, 0
    for i in range(46):
        for j in range(21):
            env.reset_n_ops()
            s, r1, r2, _, _ = env.step(state, i, j)
            if r1 != -100 and r2 != -100:
                print(state[0], s[0], r1, r2)
                a = min(a, r1)
                b = min(b, r2)
    print(a, b)
