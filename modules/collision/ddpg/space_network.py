
'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This neural network is used to compress the state matrix into a representation
into a distribution over occupied space :)

The weights of this network should be trained along with the actor and the critic.

Author: Max Ferguson
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag
from tensorflow.contrib import rnn


LEARNING_RATE = 1e-3


class SpaceNetwork:

    def __init__(self, sess, n_input, n_steps, n_hidden, output_size):
        """
        Create the network

        @n_input: The size of each input state vector
        @n_steps: The number of steps in the RNN sequence
        @n_hidden: The number of hidden units
        @n_output: The size of the output vector

        Input Size: (batch_size, n_steps, n_input)
        Output Size: (batch_size, n_mixture, 5)

        The 3rd dimension is in the form:
        [mu_x, mu_y, sigma_x, sigma_y, rho]

        Global properties
            - self.x_input: The input placeholder
            - self.y_input: The input labels placeholder
            - self.output: The output placeholder
            - self.trainable_weights: The trainable weight
        """
        self.sess = sess
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.input = tf.placeholder("float",[None, n_steps, n_input], name="input") # RNN feed

        # Build the neural network
        output, weights = self.create_network(self.input)
        self.output = output
        self.trainable_weights = weights

        # Add the optimizer code
        self.create_training_method(n_input)


    def create_training_method(self, n_input):
        #n_objects = int(n_input/2)
        #self.train_positions = tf.placeholder("float",[None, n_input], name="train_positions") # Position for training
        #self.train_labels = tf.placeholder("float",   [None, n_objects], name="train_labels") # Prob for training
        # Reshape to separate x and y
        # train_positions = tf.reshape(self.train_positions, [-1,n_objects,2])
        # Define training optimizer
        #prediction = tf.add_n([
        #    MultivariateNormalDiag(loc=mu, scale_diag=sd).prob(train_positions) for mu,sd \
        #        in zip(tf.unstack(self.mu, self.n_mixture, 1), tf.unstack(self.sd, self.n_mixture, 1))
        #])
        size = [None]+list(self.output_size)
        self.train_labels = tf.placeholder("float", size, name="train_labels") # Prob for training
        self.loss = tf.nn.l2_loss(self.output - self.train_labels)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)


    def create_network(self, inputs):
        """
        Return the weights and output tensor
        """
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Output weights and biases
        w1 = tf.Variable(tf.random_normal([self.n_hidden, 2000]))
        b1 = tf.Variable(tf.random_normal([2000]))

        w2 = tf.Variable(tf.random_normal([self.n_hidden, 80*80]))
        b2 = tf.Variable(tf.random_normal([80*80]))

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(inputs, self.n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        layer1 = tf.sigmoid(tf.matmul(outputs[-1], w1) + b1)
        layer2 = tf.sigmoid(tf.matmul(outputs[-1], w2) + b2)

        # Reshape to 80*80
        layer2 = tf.reshape(layer2, [-1,80,80])

        # Gather all the trainable weights
        weights = lstm_cell.trainable_weights + [w1, w2, b1, b2]

        # Return the weights from inside the LSTM as well as the output
        return layer2, weights


    def train_on_batch(self, x_input, train_labels):
        """
        Return a tensor that represents the loss function
        """
        loss, other = self.sess.run((self.loss, self.optimizer), feed_dict={
            self.input: x_input,
            self.train_labels: train_labels
        })
        print('Train Loss',loss)


    def test_on_batch(self, x_input, train_labels):
        """
        Return the test loss
        """
        loss = self.sess.run(self.loss, feed_dict={
            self.input: x_input,
            self.train_labels: train_labels
        })
        return loss


