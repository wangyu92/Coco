import threading
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import a3c
import env

global episode
episode = 0
EPISODES = 8000000

class CentralAgent:
    def __init__(self, num_agents=1):
        # environ = env.Environment(num_of_cluster_low=2, num_of_cluster_high=3, num_of_client_low=2, num_of_client_high=3)
        
        self.num_agents = num_agents
        
        # Global A3C model
        # self.model = a3c.A3CModel([2, environ.num_of_cluster], [environ.num_of_cluster])
        self.state_dim = (2,)
        self.action_dim = (2,)
        self.model = a3c.A3CModel(self.state_dim, self.action_dim)
        self.optimizer = [self.model.actor_optimizer(), self.model.critic_optimizer()]
        
        # Configuration tensorboard
        self.sess = tf.compat.v1.InteractiveSession()
        keras.backend.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/coco_a3c', self.sess.graph)
        
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        
        return summary_placeholders, update_ops, summary_op
        
    def train(self):
        """
        학습을 시작하는 함수.
        """
        agents = []
        for idx in range(self.num_agents):
            agent = Agent(idx, self.state_dim, self.action_dim, self.model, self.sess, self.optimizer, [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
            agents.append(agent)
            
        for idx in range(self.num_agents):
            agent = agents[idx]
            agent.start()
        
class Agent(threading.Thread):
    def __init__(self, idx, state_dim, action_dim, model, sess, optimizer, summary_ops):
        threading.Thread.__init__(self)
        
        self.idx = idx
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.central_model = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = model.discount_factor
        [self.summary_op, self.summary_placeholers, self.update_ops, self.summary_writer] = summary_ops
        
        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []
        
        # create A3C model
        self.model = a3c.A3CModel(self.state_dim, self.action_dim, self.central_model)
        
        self.avg_p_max = 0
        self.avg_loss = 0
        
        # model update interval
        self.t_max = 20
        self.t = 0
        
    def run(self):
        global episode
        
        step = 0
        state = np.zeros((2,))
        
        while episode < EPISODES:
            done = False
            dead = False
            
            score, start_life = 0, 5
            
            while not done:
                step += 1
                self.t = 1
                action, policy = self.model.get_action(state)
                
                if dead:
                    action = 0
                    dead = False
                    
                # 선택한 행동으로 한 스텝을 실행
                if action == 0:
                    next_state = state
                    next_state[0] += 1
                    reward = 1
                else:
                    next_state = state
                    next_state[1] += 1
                    reward = -1
                    
                if state[0] > 10:
                    done = True
                    
                # 정책의 최댓값
                self.avg_p_max += np.amax(self.model.actor.predict(state))
                
                score += reward
                reward = np.clip(reward, -1., 1.)
                
                # 샘플을 저장
                self.append_sample(state, action, reward)
                state = next_state
                
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0
                    
                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    print("episode:", episode, " score:", score, "step:")
                    stats = [score, self.avg_p_max / float(step), step]
                    
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholers[i]: float(stats(i))
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0
                
    def append_sample(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_dim)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)
        
    def update_local_model(self):
        self.model.actor.set_weights(self.central_model.actor.get_weights())
        self.model.critic.set_weights(self.central_model.critic.get_weights())
        
    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0
        
        if not done:
            running_add = self.model.critic.predict(self.states[-1])[0]
            
            for i in reversed(range(0, len(rewards))):
                running_add = running_add * self.discount_factor + rewards[t]
                discounted_prediction[t] = running_add
            return discounted_prediction
    
    def train_model(self, done):
        """
        Worker agent가 global network를 직접 update
        """
        discounted_prediction = self.discounted_prediction(self.rewards, done)
        
        states = np.zeros((len(self.states), 2))
        for i in range(len(self.states)):
            states[i] = self.states[i]
            
        values = self.model.critic.predict(states)
        values = np.reshape(values, len(values))
        
        advantages = discounted_predicion - values
        
        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []