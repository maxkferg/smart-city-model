
import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 300
LAYER2_SIZE = 400
LAYER3_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.00001

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,writer,state_dim,action_dim):
        self.time_step = 0
        self.sess = sess
        self.writer = writer
        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_q_network(state_dim,action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

        self.create_training_method()

        self.add_variable_summaries()

        self.summary_op = tf.summary.merge_all()

        # initialization
        self.sess.run(tf.global_variables_initializer())

        self.update_target()


    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        self.l2_loss = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.q_loss = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))
        self.cost = self.q_loss + self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)


    def create_q_network(self,state_dim,action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE
        layer3_size = LAYER3_SIZE

        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])

        W1 = self.variable([state_dim,layer1_size],state_dim)
        b1 = self.variable([layer1_size],state_dim)
        W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
        W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
        b2 = self.variable([layer2_size],layer1_size+action_dim)
        W3 = self.variable([layer2_size,layer3_size],layer2_size)
        b3 = self.variable([layer3_size],layer2_size)
        W4 = tf.Variable(tf.random_uniform([layer3_size,1], -3e-3, 3e-3))
        b4 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
        layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
        q_value_output = tf.identity(tf.matmul(layer3,W4) + b4)

        self.add_weight_summary('weights',W1,b1,W2,W2_action,b2,W3,b3,W4,b4)

        return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3,W4,b4]


    def create_target_q_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
        layer3 = tf.nn.relu(tf.matmul(layer2, target_net[5]) + target_net[6])
        q_value_output = tf.identity(tf.matmul(layer3,target_net[7]) + target_net[8])

        self.add_weight_summary('target-weights',*target_net)

        return state_input,action_input,q_value_output,target_update


    def update_target(self):
        self.sess.run(self.target_update)


    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        feed_dict = {
                self.y_input: y_batch,
                self.state_input: state_batch,
                self.action_input: action_batch
        }
        if self.time_step%100==0:
            _, summary = self.sess.run([self.optimizer, self.summary_op], feed_dict)
            self.writer.add_summary(summary, self.time_step)
        else:
            self.sess.run(self.optimizer, feed_dict)


    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]


    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })


    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})


    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


    def add_weight_summary(self, name, W1, b1, W2, W2_action, b2, W3, b3, W4, b4):
        with tf.name_scope(name):
            tf.summary.histogram("W1", W1)
            tf.summary.histogram("b1", b1)
            tf.summary.histogram("W2", W2)
            tf.summary.histogram("W2_action", W2_action)
            tf.summary.histogram("b2", b2)
            tf.summary.histogram("W3", W3)
            tf.summary.histogram("b3", b3)
            tf.summary.histogram("W4", W3)
            tf.summary.histogram("b4", b3)


    def add_variable_summaries(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('critic'):
            with tf.name_scope('loss'):
                tf.summary.scalar('loss', self.cost)

            with tf.name_scope('l2_loss'):
                tf.summary.scalar('l2_loss', self.l2_loss)

            with tf.name_scope('q_loss'):
                tf.summary.scalar('q_loss', self.q_loss)

            with tf.name_scope('y_input'):
                tf.summary.scalar('mean', tf.reduce_mean(self.y_input))
                tf.summary.scalar('max', tf.reduce_max(self.y_input))
                tf.summary.scalar('min', tf.reduce_min(self.y_input))
                tf.summary.histogram('histogram', self.y_input)

            with tf.name_scope('state_input'):
                tf.summary.scalar('mean', tf.reduce_mean(self.state_input))
                tf.summary.scalar('max', tf.reduce_max(self.state_input))
                tf.summary.scalar('min', tf.reduce_min(self.state_input))
                tf.summary.histogram('histogram', self.state_input)

            with tf.name_scope('action_input'):
                tf.summary.scalar('mean', tf.reduce_mean(self.action_input))
                tf.summary.scalar('max', tf.reduce_max(self.action_input))
                tf.summary.scalar('min', tf.reduce_min(self.action_input))
                tf.summary.histogram('histogram', self.action_input)

            with tf.name_scope('q_output'):
                tf.summary.scalar('mean', tf.reduce_mean(self.q_value_output))
                tf.summary.scalar('max', tf.reduce_max(self.q_value_output))
                tf.summary.scalar('min', tf.reduce_min(self.q_value_output))
                tf.summary.histogram('histogram', self.q_value_output)



