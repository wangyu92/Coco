import numpy as np
import tensorflow as tf
from tensorflow import keras

GAMMA = 0.99
A_DIM = 30
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
# === State ====
# 1. List of REMB*
# 2. List of number of clients*
# 3. Hardware resource usage
# 4. Bandwidth of server
# 5. Bitrate of source video


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(tf.compat.v1.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action
        self.acts = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        # Time Diffence Error
        self.act_grad_weights = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # policy loss
        action_prob = tf.reduce_sum(tf.multiply(self.out, self.acts), reduction_indices=1, keepdims=True)
        cross_entropy = tf.math.log(action_prob)
        cross_entropy = tf.multiply(cross_entropy, self.act_grad_weights)
        cross_entropy = -tf.reduce_sum(cross_entropy)
        
        # 탐색을 지속적으로 하기위한 엔트로피 오류
        entropy = tf.multiply(self.out, tf.math.log(self.out + ENTROPY_EPS))
        entropy = tf.reduce_sum(entropy)
        
        # 두 오류함수를 더해 최종 오류함수를 만듬
        self.obj =  cross_entropy + ENTROPY_WEIGHT * entropy

        # Combine the gradients here
        # gradient를 직접 계산
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.compat.v1.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):        
        with tf.compat.v1.variable_scope('actor'):
            inputs = keras.Input(shape=[self.s_dim[0], self.s_dim[1]])
            
            split_0 = keras.layers.Conv1D(128, (4), activation='relu', padding='same')(inputs[:, 0:1, :])
            split_1 = keras.layers.Conv1D(128, (4), activation='relu', padding='same')(inputs[:, 1:2, :])
            split_2 = keras.layers.Dense(128, activation='relu')(inputs[:, 2:3, -1])
            split_3 = keras.layers.Dense(128, activation='relu')(inputs[:, 3:4, -1])
            split_4 = keras.layers.Dense(128, activation='relu')(inputs[:, 4:5, -1])

            split_0_flat = keras.layers.Flatten()(split_0)
            split_1_flat = keras.layers.Flatten()(split_1)

            merge_net = keras.layers.Concatenate(axis=1)([split_0_flat, split_1_flat, split_2, split_3, split_4])
            dense_net_0 = keras.layers.Dense(1024, activation='relu')(merge_net)
            
            out = keras.layers.Dense(30, activation='linear')(dense_net_0)
            
            return inputs, out

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.compat.v1.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = keras.losses.mean_squared_error(self.out, self.td_target)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.compat.v1.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.compat.v1.variable_scope('critic'):
            inputs = keras.Input(shape=[self.s_dim[0], self.s_dim[1]])
            
            split_0 = keras.layers.Conv1D(128, (4), activation='relu', padding='same')(inputs[:, 0:1, :])
            split_1 = keras.layers.Conv1D(128, (4), activation='relu', padding='same')(inputs[:, 1:2, :])
            split_2 = keras.layers.Dense(128, activation='relu')(inputs[:, 2:3, -1])
            split_3 = keras.layers.Dense(128, activation='relu')(inputs[:, 3:4, -1])
            split_4 = keras.layers.Dense(128, activation='relu')(inputs[:, 4:5, -1])

            split_0_flat = keras.layers.Flatten()(split_0)
            split_1_flat = keras.layers.Flatten()(split_1)

            merge_net = keras.layers.Concatenate(axis=1)([split_0_flat, split_1_flat, split_2, split_3, split_4])
            dense_net_0 = keras.layers.Dense(1024, activation='relu')(merge_net)
            
            out = keras.layers.Dense(1, activation='linear')(dense_net_0)

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    
    ba_size = s_batch.shape[0]

    # 각 state에서 미래의 보상에 대해서 예측함.
    v_batch = critic.predict(s_batch)

    # 각 state에서 미래의 보상의 합까지 계산할 vector
    R_batch = np.zeros(r_batch.shape)

    # terminal : 비디오의 마지막 덩어리인지 아닌지.
    if terminal:
        # 비디오가 끝나면 마지막에는 예측할 값이 없음.
        R_batch[-1, 0] = 0  # terminal state
    else:
        # 비디오가 끝나지 않았음으로 마지막도 예측된 결과를 계산
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    # 현재 reward + discount factor * 미래의가치를 계산
    for t in reversed(range(ba_size - 1)):
        # 현재 reward + discount * 미래의 reward 예측치.
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    # TD Error ((현재 reward + discount factor * 미래예측) - BASELINE)
    # 여기서 baseline은 현재 state에서 미래에 받을 것 같은 예측치
    td_batch = R_batch - v_batch

    
    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    
    # feed를 하게되면 mean squre error를 통해 업데이트를 하게됨
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.compat.v1.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.compat.v1.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.compat.v1.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.compat.v1.summary.merge_all()

    return summary_ops, summary_vars
