import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.keras as keras
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
        outputs = keras.layers.Dense(self.env.action_shape)(dense_net_1)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Adam(0.0001)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
        
        model.summary()
        
        return model
    
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
        
        self.optimizer = keras.optimizers.Adam(0.0001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
        
        model.summary()
        
        return model
    
    @tf.function
    def train(states, returns):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss_value = self._mse(returns, predictions)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value
            
    def _mse(labels, preds):
        subs = labels - preds
        square = tf.math.pow(subs, 2)
        sums = tf.math.reduce_sum(square)
        mse = sums / labels.shape[0]
        return mse
            