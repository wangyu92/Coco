import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='2'
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import time

class Actor:
    def __init__(self, env):
        self.env = env
        self.model = self.create_model()
        
    def create_model(self):
        inputs = keras.Input(shape=(self.env.state_shape))
        hidden_remb = keras.layers.Dense(128, activation=tf.nn.relu)(inputs[:, 0:1, :])
        hidden_nu_cli = keras.layers.Dense(128, activation=tf.nn.relu)(inputs[:, 1:2, :])
        hidden_hd = keras.layers.Dense(128, activation=tf.nn.relu)(inputs[:, 2:3, :])
        
        merge_net = keras.layers.Concatenate(axis=-1)([hidden_remb, hidden_nu_cli, hidden_hd])
        
        dense_net_0 = keras.layers.Dense(1024, activation=tf.nn.relu)(merge_net)
        dense_net_1 = keras.layers.Dense(1024, activation=tf.nn.relu)(dense_net_0)
        
        # outputs = keras.layers.Dense(self.env.action_shape)(dense_net_1)
        
        # mu_sigmas = []
        mus = []
        sigmas = []
        for i in range(self.env.action_shape[0]):
            mu = keras.layers.Dense(1, activation='tanh')(dense_net_1)
            sigma = keras.layers.Dense(1, activation='relu')(dense_net_1)
            mus.append(mu)
            sigmas.append(sigma)
            
        merge_mus = keras.layers.Concatenate(axis=2)(mus)
        merge_sigmas = keras.layers.Concatenate(axis=2)(sigmas)
            
        # merge_outputs = keras.layers.Concatenate(axis=-1)(mu_sigmas)
        
        model = keras.Model(inputs=inputs, outputs=[merge_mus, merge_sigmas])
        
        self.optimizer = keras.optimizers.RMSProp(0.0001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts_actor', max_to_keep=3)
        
        # model.summary()
        
        return model
    
    def action_value(self, states):
        preds = self.model.predict(states) # list of numpy array가 반환됨
        mus = preds[0]
        sigmas = preds[1]
        
        mus = np.squeeze(mus, axis=1)
        sigmas = np.squeeze(sigmas, axis=1)
        
        mus = self._scale_up_mu(mus)
        actions = np.clip(np.random.normal(mus, sigmas), 0, None)
            
        return actions
    
    @tf.function
    def train(self, states, advantages, actions):
        states = tf.dtypes.cast(states, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            # predictions = self.model.predict(states)
            loss = self._loss(predictions, advantages, actions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
        
    @tf.function
    def _loss(self, preds, advantages, actions):
        """
        최종적으로 (batch size, ) shape을 반환하여야함.
        
        
        1. input shape을 확인한 다음. 어떻게 처리할지 확인해보자.
        
            predictions -> Tensor("predictions:0", shape=(100, 1, 60), dtype=float32)
            advantages -> Tensor("advantages:0", shape=(100,), dtype=float64)
            actions -> Tensor("actions:0", shape=(100, 30), dtype=float32)
            
        2. predictions를 이용해서 normal distribution을 구함.
        3. normal distribution에 actions를 넣어서 log prob을 구해야함.
        4. loss function 생성.
        
        """
        advantages = tf.dtypes.cast(advantages, dtype=tf.float32)
        
        mus = preds[0]
        sigmas = preds[1]
        
        mus = tf.squeeze(mus, axis=1)
        mus = self._scale_up_mu(mus)
        sigmas = tf.squeeze(sigmas, axis=1)
        
        norm_dists = tfp.distributions.Normal(mus, sigmas)
        
        logprobs = norm_dists.log_prob(actions)
        logprobs = tf.math.reduce_sum(logprobs, axis=-1)
        
        entropies = norm_dists.entropy()
        entropies = tf.math.reduce_sum(logprobs, axis=-1)
        
        return logprobs * advantages - 0.01 * entropies
                
    def _scale_up_mu(self, mu):
        return (mu + 1) * 1.5
    
    def _scale_down_up(self, mu):
        return (mu / 1.5) - 1
                
    
class Critic:
    def __init__(self, env):
        self.env = env
        self.model = self.create_model()
        
    def create_model(self):
        inputs = keras.Input(shape=(self.env.state_shape))
        hidden_remb = keras.layers.Dense(128, activation=tf.nn.relu)(inputs[:, 0:1, :])
        hidden_nu_cli = keras.layers.Dense(128, activation=tf.nn.relu)(inputs[:, 1:2, :])
        hidden_hd = keras.layers.Dense(128, activation=tf.nn.relu)(inputs[:, 2:3, :])
        
        merge_net = keras.layers.Concatenate(axis=-1)([hidden_remb, hidden_nu_cli, hidden_hd])
        
        dense_net_0 = keras.layers.Dense(1024, activation=tf.nn.relu)(merge_net)
        dense_net_1 = keras.layers.Dense(1024, activation=tf.nn.relu)(dense_net_0)
        outputs = keras.layers.Dense(1)(dense_net_1)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.optimizer = keras.optimizers.RMSProp(0.001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts_critic', max_to_keep=3)
        
        # model.summary()
        
        return model
    
    @tf.function
    def train(self, states, returns):
        states = tf.dtypes.cast(states, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            # predictions = self.model.predict(states)
            loss_value = self._mse(returns, predictions)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value
            
    def _mse(self, labels, preds):
        labels = tf.dtypes.cast(labels, dtype=tf.float32)
        preds = tf.dtypes.cast(preds, dtype=tf.float32)
        
        subs = labels - preds
        square = tf.math.pow(subs, 2)
        sums = tf.math.reduce_sum(square)
        mse = sums / labels.shape[0]
        return mse
            