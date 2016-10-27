import tensorflow as tf
import numpy as np

from netutil import *


class A3CFFNetwork(object):

    def __init__(self, state_dim, state_chn, action_dim, device='/cpu:0'):
        self._state_dim = state_dim
        self._state_chn = state_chn
        self._action_dim = action_dim
        self._device = device
        self._create_network()
        return

    def _create_network(self):
        state_dim = self._state_dim
        state_chn = self._state_chn
        action_dim = self._action_dim
        with tf.device(self._device):
            # state input
            self.s = tf.placeholder('float', [None, state_dim, state_dim, state_chn])

            # conv1
            self.W_conv1 = weight_variable([8, 8, state_chn, 16])
            self.b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 4) + self.b_conv1)

            # conv2
            self.W_conv2 = weight_variable([4, 4, 16, 32])
            self.b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            h_conv2_out_size = np.prod(h_conv2.get_shape().as_list()[1:])
            print 'h_conv2_out_size', h_conv2_out_size
            h_conv2_flat = tf.reshape(h_conv2, [-1, h_conv2_out_size])

            # fc1
            self.W_fc1 = weight_variable([h_conv2_out_size, 256])
            self.b_fc1 = bias_variable([256])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # fc2: (pi) for policy output
            self.W_fc2 = weight_variable([256, action_dim])
            self.b_fc2 = bias_variable([action_dim])
            self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)

            # fc3: (v)  for value output
            self.W_fc3 = weight_variable([256, 1])
            self.b_fc3 = bias_variable([1])
            v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
            self.value_output = tf.reshape(v_, [-1])

        return

    def prepare_loss(self, entropy_beta):
        # taken action (input for policy)
        self.a = tf.placeholder('float', [None, self._action_dim])
        # temporary difference (R-V)  (input for policy)
        self.td = tf.placeholder('float', [None])

        # avoid NaN
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
        # policy entropy
        entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
        # policy loss L = log pi(a|s, theta) * (R - V)
        # (Adding minus, because the original paper's objective function is for gradient ascent,
        # but we use gradient descent optimizer.)
        policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.mul(log_pi, self.a),
                                                   reduction_indices=1) * self.td + entropy * entropy_beta)

        # R (input for value)
        self.r = tf.placeholder('float', [None])
        # value loss (output) L = (R-V)^2
        # value_loss = tf.reduce_mean(tf.square(self.r - self.value_output))
        value_loss = 0.5 * tf.nn.l2_loss(self.r - self.value_output)
        self.total_loss = policy_loss + value_loss
        return

    def get_total_loss(self):
        return self.total_loss

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3
        ]

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, 'A3CFFNetwork') as scope:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_ops.append(tf.assign(dst_var, src_var))
                return tf.group(*sync_ops, name=scope)

    def run_policy_and_value(self, sess, state):
        policy, value = sess.run([self.pi, self.value_output], feed_dict={self.s: [state]})
        return policy[0], value[0]

    def run_policy(self, sess, state):
        policy = sess.run(self.pi, feed_dict={self.s: [state]})
        return policy[0]

    def run_value(self, sess, state):
        value = sess.run(self.value_output, feed_dict={self.s: [state]})
        return value[0]


if __name__ == '__main__':
    net = A3CFFNetwork(84, 3, 2)
    net.create_loss(0.01)
    print 'a3c_network.py'
