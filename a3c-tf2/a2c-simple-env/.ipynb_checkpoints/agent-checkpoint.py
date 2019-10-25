import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms
        self.params = {
            'value': 0.5,
            'entropy': 0.001,
            'gamma':0.99}
        self.model = model
        
    def train(self, env, batch_size=32, updates=500):
        # storage helpers for a single batch of data
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        states = np.empty((batch_size, ) + env.state.shape)
        
        # training loop: collect samples, send to optimize, repeat updates times
        ep_rews = [0.0]
        next_state = env.reset()
        
        for update in range(updates):
            # batch_size 만큼 시도해보면서 데이터를 모음
            
            print('.', end='')
            if update % 100 == 0 and update != 0:
                print()
            
            for step in range(batch_size):
                states[step] = next_state.copy()
                _, actions[step], values[step] = self.model.action_value(np.expand_dims(next_state, axis=0))
                next_state, rewards[step], dones[step] = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_state = env.reset()

            
            _, _, next_value = self.model.action_value(next_state[None, :])
            
#             print(actions)
            returns, advs = self.model._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
        
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it

            losses = self.model.train_on_batch(states, [acts_and_advs, returns])
        return ep_rews

    def test(self, env):
        # unchanged from previous section
        state, done, ep_reward = env.reset(), False, 0
        while not done:
            _, action, _ = self.model.action_value(state[None, :])
            state, reward, done = env.step(action)
            ep_reward += reward
        return ep_reward