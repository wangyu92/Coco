import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class A2CAgent:
    def __init__(self, model):
        self.model = model
        self.params = {
            'value': 0.5,
            'entropy': 0.001,
            'gamma':0.99
        }
        
    def test(self, env):
        state, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(state[None, :])
            state, reward, done = env.step(action)
            ep_reward += reward
        return ep_reward
    
    def train(self, env, batch_size=32, updates=100):
        # storage helpers for a single batch of data
        actions = np.empty((batch_size,), dtype=np.float32)
        rewards, dones, values = np.empty((3, batch_size))
        states = np.empty((batch_size, ) + env.state.shape)
        
        # training loop: collect samples, send to optimize, repeat updates times
        ep_rews = [0.0]
        next_state = env.reset()
        for update in range(updates):
            if update % 100 == 0:
                print(str(update) + ' epochs')
                
            # batch_size 만큼 시도해보면서 데이터를 모음
            for step in range(batch_size):
                states[step] = next_state.copy()
                actions[step], values[step] = self.model.action_value(np.expand_dims(next_state, axis=0))
                next_state, rewards[step], dones[step] = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_state = env.reset()

            
            _, next_value = self.model.action_value(next_state[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(states, [advs, returns])
        return ep_rews
    
    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages