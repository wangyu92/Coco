import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras

class A2C(tf.keras.Model):
    def __init__(self, env, learning_rate=0.0007):
        super().__init__('mlp_policy')
        
        self.env = env
        
        print("learning_rate = %f" % (learning_rate))
        self.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss = [self.loss_policy, self.loss_value],
            run_eagerly=True
        )
        
        # actor
        self.hidden_p1 = keras.layers.Dense(128, activation='relu')
        self.hidden_p2 = keras.layers.Dense(128, activation='relu')
        self.mu = keras.layers.Dense(1, activation='tanh')
        self.sigma = keras.layers.Dense(1, activation='softplus')
        
        # critic
        self.hidden_v1 = keras.layers.Dense(128, activation='relu')
        self.hidden_v2 = keras.layers.Dense(128, activation='relu')
        self.value = keras.layers.Dense(1, name='value')
        
    def call(self, inputs):
        # inputs is a numpy array, convert to tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.int64)
        
        # actor
        hidden_p = self.hidden_p1(x)
        hidden_p = self.hidden_p2(hidden_p)
        mu = self.mu(hidden_p)
        sigma = self.sigma(hidden_p)
        self.norm_dist = tfp.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(self.norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var, self.env.action_low, self.env.action_high)
        
        # critic
        hidden_v = self.hidden_v1(x)
        hidden_v = self.hidden_v2(hidden_v)
        out_value = self.value(hidden_v)
    
        return action_tf_var, out_value
    
    def action_value(self, obs):
        policy, value = self.predict(obs)

        return np.squeeze(policy, axis=-1), np.squeeze(value, axis=-1)
    
    
    def loss_policy(self, advantages, action):
        logprobs = self.norm_dist.log_prob(action)
        entropy = self.norm_dist.entropy()
        return -tf.math.reduce_mean(logprobs * advantages - 0.01 * entropy)
    
    def loss_value(self, returns, value):
        return keras.losses.mean_squared_error(returns, value)
    
    def load_model(self, name):
        """
        해당 클래스 자체가 모델이기 때문에 actor와 critic으로 나누어서 저장하지 않아도 될 것으로 보임.
        """
        self.load_weights(name + ".h5")
        
    def save_model(self, name):
        """
        해당 클래스 자체가 모델이기 때문에 actor와 critic으로 나누어서 저장하지 않아도 될 것으로 보임.
        """
        self.save_weights(name + ".h5")
        