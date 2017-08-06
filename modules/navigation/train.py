import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ddpg import DDPG, REPLAY_START_SIZE
#from exploration import GuidedExploration
from environment import LearningEnvironment
from debug import *


PATH = 'results/checkpoints'
LOGS = 'results/logs'
EPOCHS = 1000
EPISODES = 100
PARTICLES = 5
RENDER = True
STEP = 4


def fill_buffer(env, agent, epsilon):
    while agent.replay_buffer.count() <=  REPLAY_START_SIZE:
        done = False
        state = env.reset()
        rewards = 0
        # Training
        while not done:
            action = agent.noise_action(state, epsilon)
            next_state, reward, done, info = env.step(action, STEP)
            agent.perceive(state, action, reward, next_state, done)
            # Setup for next cycle
            state = next_state
            rewards += reward


def train(env, agent, epsilon):
    done = False
    state = env.reset()
    rewards = 0
    # Training
    while not done:
        action = agent.noise_action(state, epsilon)
        next_state, reward, done, info = env.step(action, STEP)
        agent.perceive(state, action, reward, next_state, done)
        # Setup for next cycle
        state = next_state
        rewards += reward
    return rewards


def test(env, agent, render=False):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        action = agent.action(state)
        next_state, reward, done, info = env.step(action, STEP)
        state = next_state
        rewards += reward
        if render:
            env.background = get_q_background(env, agent, action)
            env.render()
    return rewards




if __name__=='__main__':
    # Setup
    epsilon = 0.186
    disabled = not RENDER
    env = LearningEnvironment(num_particles=PARTICLES, disable_render=disabled)
    writer = tf.summary.FileWriter(LOGS, graph=tf.get_default_graph())
    agent = DDPG(env,writer)
    agent.restore_model(PATH)

    # Fill the buffer
    fill_buffer(env, agent, epsilon)

    # Train on a large number of epochs
    for epoch in range(EPOCHS):
        print("\nEPOCH: {0} epsilon={1:.3f}".format(epoch,epsilon))
        rewards = []

        # Run a few episodes
        for episode in tqdm(range(EPISODES)):
            reward = train(env, agent, epsilon)
            rewards.append(reward)

        # Evaluate
        train_reward = np.mean(rewards)

        test_reward = np.mean([test(env, agent) for i in range(20)])
        print("Train Reward {0}, Test Reward {1}".format(train_reward, test_reward))

        # Render
        test(env, agent, render=RENDER)

        # Save model
        agent.save_model(PATH,epoch)

        # Update parameters
        epsilon *= 0.995





