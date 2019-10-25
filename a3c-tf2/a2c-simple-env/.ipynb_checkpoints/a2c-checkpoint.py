import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.params = {
            'value': 0.5,
            'entropy': 0.001,
            'gamma':0.99}
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = keras.layers.Dense(128, activation='relu')
        self.hidden2 = keras.layers.Dense(128, activation='relu')
        self.value = keras.layers.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = keras.layers.Dense(num_actions, activation='softmax', name='policy_logits')
        self.dist = ProbabilityDistribution()
        
        self.compile(
            optimizer=keras.optimizers.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    # 레이어를 구성해주는 부분인 것 같음
    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)

        return np.squeeze(logits, axis=0), np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
    
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

    def _policy_loss(self, actions, advantages, pred):
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
    
    @tf.function
    def train_manually(states, actions, rewards, values, dones, next_value):
        with tf.GradientTape() as tape:
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            pred_policies, pred_values = self(states, training=True)
            loss = self._policy_loss(actions, advantages, pred_policies)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))