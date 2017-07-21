import tensorflow as tf
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LAYER3_SIZE = 200
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,compression_network,state_dim,action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.compression_network = compression_network
        # create actor network
        self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

    def create_network(self,state_dim,action_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE
        layer3_size = LAYER3_SIZE

        state_input = self.compression_network.input
        compressed_input = self.compression_network.output

        W1 = self.variable([state_dim,layer1_size],state_dim)
        b1 = self.variable([layer1_size],state_dim)
        W2 = self.variable([layer1_size,layer2_size],layer1_size)
        b2 = self.variable([layer2_size],layer1_size)
        W3 = self.variable([layer2_size,layer3_size],layer2_size)
        b3 = self.variable([layer3_size],layer2_size)
        W4 = tf.Variable(tf.random_uniform([layer3_size,action_dim],-3e-3,3e-3))
        b4 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

        layer1 = tf.nn.relu(tf.matmul(compressed_input,W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
        layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
        action_output = tf.tanh(tf.matmul(layer3,W4) + b4)

        # Collect all the weights
        weights = [W1,b1,W2,b2,W3,b3,W4,b4] + self.compression_network.trainable_weights

        return state_input, action_output, weights

    def create_target_network(self,state_dim,action_dim,net):
        state_input = self.compression_network.input
        compressed_input = self.compression_network.output

        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(compressed_input,target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
        layer3 = tf.nn.relu(tf.matmul(layer2,target_net[4]) + target_net[5])
        action_output = tf.tanh(tf.matmul(layer3,target_net[6]) + target_net[7])

        return state_input,action_output,target_update,target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,state):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:[state]
            })[0]


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch
            })

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''


