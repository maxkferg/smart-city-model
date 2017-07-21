
import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 200
LAYER2_SIZE = 100
LAYER3_SIZE = 100
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,compression_network,state_dim,action_dim):
        self.time_step = 0
        self.sess = sess
        self.compression_network = compression_network
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

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

    def create_q_network(self,state_dim,action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE
        layer3_size = LAYER3_SIZE

        state_input = self.compression_network.input
        compressed_input = self.compression_network.output
        action_input = tf.placeholder("float",[None,action_dim])

        W1 = self.variable([state_dim,layer1_size],state_dim)
        b1 = self.variable([layer1_size],state_dim)
        W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
        W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
        b2 = self.variable([layer2_size],layer1_size+action_dim)
        W3 = self.variable([layer2_size,layer3_size],layer2_size)
        b3 = self.variable([layer3_size],layer2_size)
        W4 = tf.Variable(tf.random_uniform([layer3_size,1],-3e-3,3e-3))
        b4 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

        layer1 = tf.nn.relu(tf.matmul(compressed_input, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
        layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
        q_value_output = tf.identity(tf.matmul(layer3,W4) + b4)

        # Collect all the weights
        weights = [W1,b1,W2,W2_action,b2,W3,b3,W4,b4] + self.compression_network.trainable_weights

        return state_input,action_input,q_value_output,weights

    def create_target_q_network(self,state_dim,action_dim,net):
        state_input = self.compression_network.input
        compressed_state = self.compression_network.output
        action_input = tf.placeholder("float",[None,action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(compressed_state,target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
        layer3 = tf.nn.relu(tf.matmul(layer2,target_net[5]) + target_net[6])
        q_value_output = tf.identity(tf.matmul(layer3,target_net[7]) + target_net[8])

        return state_input,action_input,q_value_output,target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,y_batch,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch
            })

    def gradients(self,state_batch,action_batch):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch
            })[0]

    def target_q(self,state_batch,action_batch):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''