
'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This neural network is used to compress the state matrix into a representation
that is useful for the actor and the critic.

The weights of this network should be trained along with the actor and the critic.

Author: Max Ferguson
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn


class CompressionNetwork:

    def __init__(self, n_input, n_steps, n_hidden, n_output):
        """
        Create the network

        @n_input: The size of each input state vector
        @n_steps: The number of steps in the RNN sequence
        @n_hidden: The number of hidden units
        @n_output: The size of the output vector

        Input Size: (batch_size, n_steps, n_input)
        Output Size: (batch_size, n_output)

        Global properties
            - self.input: The input placeholder
            - self.output: The output placeholder
            - self.trainable_weights: The trainable weight
        """
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.input = tf.placeholder("float", [None, n_steps, n_input])

        output, weights = self.create_network(self.input)
        self.output = output
        self.trainable_weights = weights


    def create_network(self, inputs):
        """
        Return the weights and output tensor
        """
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Output weights and biases
        w = tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
        b = tf.Variable(tf.random_normal([self.n_output]))

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(inputs, self.n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        output = tf.matmul(outputs[-1], w) + b

        # Gather all the trainable weights
        weights = lstm_cell.trainable_weights + [w,b]

        # Return the weights from inside the LSTM as well as the output
        return output, weights

