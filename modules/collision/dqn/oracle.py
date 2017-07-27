
'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This neural network is used to compress the state matrix into a representation
that is useful for the actor and the critic.

The weights of this network should be trained along with the actor and the critic.

Author: Max Ferguson
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
from tensorflow.contrib.seq2seq import Helper, TrainingHelper, BasicDecoder
from tensorflow.contrib.layers import xavier_initializer
from tfmodels.seq2seq import TrainableSequence2Sequence



class OracleNetwork(TrainableSequence2Sequence):



    def create_reward(self, state, reuse):
        """
        Create a deep NN that estimates collisions from state
        """
        state_size = state.get_shape().as_list()[1]
        n_positions = int(state_size/2)
        n_particles = int(state_size/4)
        n_output = 1

        xavier = xavier_initializer()

        with tf.variable_scope("reward", reuse=reuse):
            w1 = tf.get_variable("w1", (state_size, n_positions), initializer=xavier)
            b1 = tf.get_variable("b1", (n_positions,), dtype=tf.float32)

            w2 = tf.get_variable("w2", (n_positions, n_particles), initializer=xavier)
            b2 = tf.get_variable("b2", (n_particles,), dtype=tf.float32)

            w3 = tf.get_variable("w3", (n_particles, n_particles), initializer=xavier)
            b3 = tf.get_variable("b3", (n_particles,), dtype=tf.float32)

            w4 = tf.get_variable("w4", (n_particles, n_output), initializer=xavier)
            b4 = tf.get_variable("b4", (n_output,), dtype=tf.float32)

        layer1 = tf.square(tf.matmul(state,w1) + b1)  # Calculate element-wise x,y distances
        layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2) # Calculate remaining space between particles
        layer3 = tf.nn.sigmoid(tf.matmul(layer2,w3) + b3) # Calculate the number of collisions
        return tf.matmul(layer3,w4) + b4 # Return the total reward based on number of collisions

