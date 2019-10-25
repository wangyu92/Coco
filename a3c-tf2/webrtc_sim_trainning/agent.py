import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.keras as keras
import numpy as np
import time
import datetime

from env import Env
from model import Actor, Critic

class Agent:
    def __init__(self):
        self.env = Env()
        self.actor = Actor(env)
        self.critic = Critic(env)
        
        self.params = {
            'gamma':0.99,
            'epsilon':0.2
        }
        
    def config_tensorboard(self):
        # -- tensorboard --
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape' + current_time + 'train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_log_dir = 'logs/gradient_tape' + current_time + 'test'
        self.test_summary_wrtier = tf.summary.create_file_writer(test_log_dir)
        
        # -- metric --
        loss_metric = keras.metrics.Mean(name='train_loss')
        
    def train(batch_size=100, updates=10000):
        self.config_tensorboard()
        
        actions = np.empty((batch_size,) + env.action_shape, dtype=np.float32)
        rewards, dones, values = np.empty((3, batch_size))
        states = np.empty((batch_size,) + env.state_shape)
        
        reward_epi = [0.]
        next_state = env.reset()
        
        for update in range(updates):
            
            # 한 정책에서 trajectory를 따라서 쭉 수행 및 기록.
            for step in range(batch_size):
                states[step] = next_state.copy()
                actions[step] = self.actor.model(np.expand_dims(next_state, axis=0), training=True)
                values[step] = self.critic.model(np.expand_dims(next_state, axis=0), training=True)
                
                next_state, rewards[step], dones[step] = env.step(actions[step])
                
                reward_epi[-1] += rewards[step]
                if dones[step]:
                    reward_epi.append(0.0)
                    next_state = env.reset()
                    
            next_value = self.critic.model(np.expand_dims(next_state, axis=0), training=True)
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            
            ### training ###
            # actor network
            
            
            # critic network
            loss_value = critic.train(states, returns)
            
            # save checkpoint
            self.critic.ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = self.critic.ckpt_manager.save()
                
            # tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss_value', loss_value, step=update)
                    
    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        
        advantages = returns - values
        return returns, advantages
        
    