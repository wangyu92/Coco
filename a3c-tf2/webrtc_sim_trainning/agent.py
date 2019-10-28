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
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)
        
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
        
    def restore_checkpoint(self):
        is_exist_checkpoint = len(self.actor.ckpt_manager.checkpoints) != 0
        if is_exist_checkpoint:
            lastest_checkpoint = self.actor.ckpt_manager.checkpoints[-1]
            self.actor.ckpt.restore(lastest_checkpoint)
            print("Actor : Restored from {}".format(lastest_checkpoint))
        else:
            print("Actor : Initializing from scrach.")
            
        is_exist_checkpoint = len(self.critic.ckpt_manager.checkpoints) != 0
        if is_exist_checkpoint:
            lastest_checkpoint = self.critic.ckpt_manager.checkpoints[-1]
            self.critic.ckpt.restore(lastest_checkpoint)
            print("Critic : Restored from {}".format(lastest_checkpoint))
        else:
            print("Critic : Initializing from scrach.")
        
    def train(self, batch_size=100, updates=10000):
        self.config_tensorboard()
        self.restore_checkpoint()
        
        actions = np.empty((batch_size,) + self.env.action_shape, dtype=np.float32)
        rewards, dones, values = np.empty((3, batch_size))
        states = np.empty((batch_size,) + self.env.state_shape)
        
        reward_epi = [0.]
        next_state = self.env.reset()
        
        for update in range(updates):
            
            # 한 정책에서 trajectory를 따라서 쭉 수행 및 기록.
            for step in range(batch_size):
                states[step] = next_state.copy()
                sel_action = self.actor.action_value(np.expand_dims(next_state, axis=0))
                actions[step] = np.squeeze(sel_action, axis=0)
                values[step] = np.squeeze(self.critic.model(np.expand_dims(next_state, axis=0), training=True), axis=0)
                
                next_state, rewards[step], dones[step] = self.env.step(actions[step])
                
                reward_epi[-1] += rewards[step]
                if dones[step]:
                    reward_epi.append(0.0)
                    next_state = env.reset()
                    
            next_value = self.critic.model(np.expand_dims(next_state, axis=0), training=True)
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            
            ### training ###
            # actor network
            loss_policy = self.actor.train(states, advantages, actions)
            
            # critic network
            loss_value = self.critic.train(states, returns)
            
            # save checkpoint
            self.actor.ckpt.step.assign_add(1)
            self.critic.ckpt.step.assign_add(1)
            if int(self.actor.ckpt.step) % 10 == 0:
                save_path_actor = self.actor.ckpt_manager.save()
                save_path_critic = self.critic.ckpt_manager.save()
                
            # tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('advantages', np.mean(advantages), step=update)
                tf.summary.scalar('loss_policy', np.mean(loss_policy), step=update)
                tf.summary.scalar('loss_value', np.mean(loss_value), step=update)
                tf.summary.scalar('rewards_epi', sum(reward_epi), step=update)
                    
    def _returns_advantages(self, rewards, dones, values, next_value):
        next_value = tf.squeeze(next_value, axis=[0, 1])
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        
        advantages = returns - values
        return returns, advantages
        
    