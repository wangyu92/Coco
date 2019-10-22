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
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
        
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
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
        
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it

            losses = self.model.train_on_batch(states, [acts_and_advs, returns])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        """
        return = Q
        value = V
        """
        
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def test(self, env):
        # unchanged from previous section
        state, done, ep_reward = env.reset(), False, 0
        while not done:
            _, action, _ = self.model.action_value(state[None, :])
            state, reward, done = env.step(action)
            ep_reward += reward
        return ep_reward

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * keras.losses.mean_squared_error(returns, value)
    
    def _logits_loss(self, acts_and_advs, pred):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        
        # 실제로 한 행동을 onehoe encoding하여 수행
        actions_onehot = tf.one_hot(tf.cast(actions, tf.int32), 2, dtype=tf.float32)
        
        # 학습이 재대로 안되었던 것은 결굴 shape의 문제였음.
        # output의 shape을 잘 확인하는 것이 중요.
        actions_onehot = tf.reshape(actions_onehot, shape=[32, 2])
        advantages = tf.reshape(advantages, shape=[32])
        
        responsible_outputs = tf.math.reduce_sum(pred * actions_onehot, axis=[1])
        
        # softmax를 사용하면 이 부분이 필요하지 않음.
        # responsible_outputs = tf.clip_by_value(responsible_outputs, 0, tf.float32.max)
        
        # 로그에 음수값이 들어가서 nan이 발생되어 학습이 안되었음.
        # 이부분을 clip을 통해 해결함.
        policy_loss = tf.math.log(responsible_outputs) * advantages
        policy_loss = -tf.math.reduce_sum(policy_loss)
#         print(policy_loss)
        
        entropy_loss = tf.math.reduce_sum(pred * tf.math.log(pred), axis=[1])
#         print(entropy_loss)

#         print(policy_loss - self.params['entropy'] * entropy_loss)
        
        # 결국 최종적으로 [32,] 형태의 loss가 반환되어야함.
        return policy_loss - self.params['entropy'] * entropy_loss
    
    def _logits_loss2(self, acts_and_advs, pred):
        """
        결국 최종적으로 (32,) shape의 tensor가 반횐된어야함. 즉, batch에서 각 step마다 loss function이 있는 것임.
        """
        
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, pred, sample_weight=advantages)
        print(policy_loss)
        
        # entropy loss can be calculated via CE over itself
        entropy_loss = keras.losses.categorical_crossentropy(pred, pred, from_logits=True)
        print(entropy_loss)
        # here signs are flipped because optimizer minimizes
        
#         print(policy_loss - self.params['entropy'] * entropy_loss)
        return policy_loss - self.params['entropy'] * entropy_loss