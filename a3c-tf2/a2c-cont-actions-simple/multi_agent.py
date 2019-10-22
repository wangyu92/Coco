from multiprocessing import Process, Queue
import threading
import time
import numpy as np

from env import Env
from a2c import A2C

import tensorflow as tf
import tensorflow.keras as keras

class MultiAgent:
    def __init__(self, num_agents=16, batch_size=32, updates=500):
        self.num_agents = num_agents
        self.stime = time.time()
        
        self.env = Env()
        self.model = A2C(self.env, learning_rate=0.001)
        
        self.batch_size = batch_size
        self.updates = updates
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

    def train(self):
        # 프로세스간 통신을 위해서 큐 생성
        end_indicate_queues = []

        for i in range(self.num_agents):
            end_indicate_queues.append(Queue(1))

        # create a coordinator and multiple agent processes
        # (note: threading is not desirable due to python GIL)
        coordinator = Process(target=self.central, args=(end_indicate_queues,))
        coordinator.start()

        agents = []
        for i in range(self.num_agents):
            agents.append(Process(target=self.worker, args=(i, end_indicate_queues[i])))

        for i in range(self.num_agents):
            agents[i].start()
            
        for i in range(self.num_agents):
            agents[i].join()

        coordinator.join()
        
        self.print_log('Central', 'All thread ends')
        
    def central(self, end_indicate_queues):
        pass
    
    def worker(self, idx, end_indicate_queue):
        self.print_log('Thread %02d' % idx, 'thread started')
    
        self.env.reset()
        
        actions = np.empty((self.batch_size,), dtype=np.float32)
        rewards, dones, values = np.empty((3, self.batch_size))
        states = np.empty((self.batch_size,) + self.env.state.shape)
        
        # training loop: collect smaples, send to optimize, repeat updates times
        ep_rews = [0.0]
        next_state = self.env.reset()
        
        for update in range(self.updates):
            if update % 50 == 0 and update != 0:
                self.print_log('Thread %02d' % idx, str(update) + ' epoches')
            
            for step in range(self.batch_size):
                states[step] = next_state.copy()
                actions[step], values[step] = self.model.action_value(np.expand_dims(next_state, axis=0))
                next_state, rewards[step], dones[step] = self.env.step(actions[step])
                
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_state = self.env.reset()
                    
            _, next_value = self.model.action_value(next_state[None, :])
            returns, advs = self.compute_returns_and_advantages(rewards, dones, values, next_value)
            loess = self.model.train_on_batch(states, [advs, returns])
            
        end_indicate_queue.put(True)
            
    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages
    
    def print_log(self, sender, msg):
        print("[%s, %f] %s" % (sender, time.time() - self.stime, msg))
