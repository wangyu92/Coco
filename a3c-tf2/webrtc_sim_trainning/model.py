import os, sys
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import time

class Actor:
    def __init__(self, env):
        keras.backend.set_floatx('float64')
        
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
            sigma = keras.layers.Dense(1, activation='softplus')(dense_net_1)
            mus.append(mu)
            sigmas.append(sigma)
            
        merge_mus = keras.layers.Concatenate(axis=2)(mus)
        merge_sigmas = keras.layers.Concatenate(axis=2)(sigmas)
            
        # merge_outputs = keras.layers.Concatenate(axis=-1)(mu_sigmas)
        
        model = keras.Model(inputs=inputs, outputs=[merge_mus, merge_sigmas])
        
        self.optimizer = keras.optimizers.Adam(0.0001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts_actor', max_to_keep=3)
        
        # model.summary()
        
        return model
    
    def action_value(self, states):
        preds = self.model.predict(states) # list of numpy array가 반환됨
        mus = preds[0]
        sigmas = preds[1] + 10
        
        mus = np.squeeze(mus, axis=1)
        sigmas = np.squeeze(sigmas, axis=1)
        
        mus = self._scale_up_mu(mus)
        actions = np.clip(np.random.normal(mus, sigmas), 0, None)
            
        return actions
    
    @tf.function
    def train(self, states, advantages, actions):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            # predictions = self.model.predict(states)
            loss = self._loss(predictions, advantages, actions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
        
    @tf.function
    def _loss(self, preds, advantages, actions):
        mus = preds[0]
        sigmas = preds[1]
        
        mus = tf.squeeze(mus, axis=1)
        mus = self._scale_up_mu(mus)
        sigmas = tf.squeeze(sigmas, axis=1) + 10
        
        norm_dists = tfp.distributions.Normal(mus, sigmas)
        
        logprobs = norm_dists.log_prob(actions)
        logprobs = tf.math.reduce_sum(logprobs, axis=-1)
        
        entropies = norm_dists.entropy()
        entropies = tf.math.reduce_sum(entropies, axis=-1)
        
        return -logprobs * advantages - 0.01 * entropies
                
    def _scale_up_mu(self, mu):
        return (mu + 1) * 1500
    
    def _scale_down_up(self, mu):
        return (mu / 1500) - 1
                
    
class Critic:
    def __init__(self, env):
        keras.backend.set_floatx('float64')
        
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
        
        self.optimizer = keras.optimizers.Adam(0.001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts_critic', max_to_keep=3)
        
        # model.summary()
        
        return model
    
    @tf.function
    def train(self, states, returns):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            # predictions = self.model.predict(states)
            loss_value = self._mse(returns, predictions)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value
            
    def _mse(self, labels, preds):
        subs = labels - preds
        square = tf.math.pow(subs, 2)
        sums = tf.math.reduce_sum(square)
        mse = sums / labels.shape[0]
        return mse
            