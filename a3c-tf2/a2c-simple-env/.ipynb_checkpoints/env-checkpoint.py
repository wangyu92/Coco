import numpy as np

class Env:
    def __init__(self):
        self.action_dim = 2
        self.state = np.array([1., 1.])
    
    def reset(self):
        self.state = np.array([1., 1.])
        return self.state
        
    def obs(self):
        return np.array(self.state)
    
    def step(self, action):
        # 아래로 이동
        if action == 0:
            self.state[0] += 1
        # 오른쪽으로 이동
        elif action == 1:
            self.state[1] += 1
        reward, done = self.calc_reward()
        return self.state, reward, done
            
    def calc_reward(self):
        if self.state[0] > 3 or self.state[1] > 3:
            return -1, True
        elif self.state[0] == 1 and self.state[1] == 3:
            return -1, False
        elif self.state[0] == 3 and self.state[1] == 1:
            return -1, False
        elif self.state[0] == 3 and self.state[1] == 3:
            return 1, True
        else:
            return 1, False