import numpy as np
import tensorflow as tf
import tflearn


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
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        # TD Error
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # policy loss
        a = tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts), reduction_indices=1, keep_dims=True))
        a = tf.multiply(a, -self.act_grad_weights)
        a = tf.reduce_sum(a)
        
        # 탐색을 지속적으로 하기위한 엔트로피 오류
        entropy = tf.multiply(self.out, tf.log(self.out + ENTROPY_EPS))
        entropy = tf.reduce_sum(entropy)
        
        # 두 오류함수를 더해 최종 오류함수를 만듬
        self.obj =  a + ENTROPY_WEIGHT * entropy

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            # list of remb
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')
            # list of number of clients
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')
            # hardware resource usage
            split_2 = tflearn.fully_connected(inputs[:, 2:3, -1], 128, activation='relu')
            # bandwidth of server
            split_3 = tflearn.fully_connected(inputs[:, 3:4, -1], 128, activation='relu')
            # bitrate of source video
            split_4 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)

            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2, split_3, split_4], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 1024, activation='relu')
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='linear')

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
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            # list of remb
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')
            # list of number of clients
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')
            # hardware resource usage
            split_2 = tflearn.fully_connected(inputs[:, 2:3, -1], 128, activation='relu')
            # bandwidth of server
            split_3 = tflearn.fully_connected(inputs[:, 3:4, -1], 128, activation='relu')
            # bitrate of source video
            split_4 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)

            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2, split_3, split_4], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

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

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
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
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
