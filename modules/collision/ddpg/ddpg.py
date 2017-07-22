# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Max Ferguson
# Date: 2017.7.28
# -----------------------------------
import os
import gym
import tensorflow as tf
import numpy as np
from .ou_noise import OUNoise
from .compression_network import CompressionNetwork
from .critic_network import CriticNetwork
from .actor_network import ActorNetwork
from .replay_buffer import ReplayBuffer

# Hyper Parameters:
CN_HIDDEN = 40  # Twice the output size
CN_OUTPUT = 20  # 4 properties, 10 particles

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.8


class DDPG:
    """docstring for DDPG"""
    def __init__(self, env):
        self.name = 'DDPG' # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks

        # Calculate the compression network size
        cn_steps = env.observation_space.shape[0]
        cn_input = env.observation_space.shape[1]
        cn_hidden = CN_HIDDEN
        cn_output = CN_OUTPUT

        self.state_dim = cn_output
        self.action_dim = np.prod(env.action_space.shape)

        self.sess = tf.InteractiveSession()

        self.cn = CompressionNetwork(cn_input, cn_steps, cn_hidden, cn_output)
        self.actor_network = ActorNetwork(self.sess, self.cn, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.cn, self.state_dim, self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        self.saver = tf.train.Saver()

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def save_model(self, path, episode):
        filename = os.path.join(path,"model.ckpt")
        self.saver.save(self.sess, filename, episode)
        print("Saved model to ",filename)

    def restore_model(self, path):
        try:
            checkpoint = tf.train.latest_checkpoint(path)
            self.saver.restore(self.sess, checkpoint)
            print("Restored model from ",checkpoint)
        except Exception as e:
            print(e)

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        return self.action(state) + self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()











