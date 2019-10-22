import numpy as np

class Env:
    """
    그리드월드를 생성하는데 discrete한 action아니라 continuous한 action으로
    방향을 결정하도록 하면 될듯.
    """
    
    def __init__(self):
        self.action_dim = 1
        self.action_low = 0
        self.action_high = 10
    
    def reset(self):
        self.state = np.array([1, 1])
        return self.state
        
    def obs(self):
        return self.state
        
    def step(self, action):
        next_state = self.state.copy()
        
        # action에 따라서 방향을 결정하고 다음 state를 생성함.
        # 아래
        if action < 5.0:
            next_state[0] += 1
            
        # 오른쪽
        elif action >= 5.0:
            next_state[1] += 1
        
        reward = 0
        done = False
        
        # state는 3 x 3 그리드 월드이고 [3, 3]으로 가는 것이 최종 목표임
        if next_state[0] < 1 or next_state[0] > 3:
            reward = -10
            done = True
        elif next_state[1] < 1 or next_state[1] > 3:
            reward = -10
            done = True
        elif next_state[0] == 2 and next_state[1] == 2:
            reward = -10
            done = True
        elif next_state[0] == 3 and next_state[1] == 3:
            reward = 10
            done = True
        else:
            reward = 1
            
        self.state = next_state
        return next_state, reward, done