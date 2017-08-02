import numpy as np
from tqdm import tqdm
from ddpg import DDPG
#from exploration import GuidedExploration
from environment import LearningEnvironment

PATH = 'results/ddpg'
EPOCHS = 1000
EPISODES = 100
RENDER = False


def train(env, agent, epsilon):
    done = False
    state = env.reset()
    rewards = 0
    # Training
    while not done:
        action = agent.noise_action(state, epsilon)
        next_state, reward, done, info = env.step(action)
        agent.perceive(state, action, reward, next_state, done)
        # Setup for next cycle
        state = next_state
        rewards += reward
    return rewards


def test(env, agent, render=False):
    done = False
    state = env.reset()
    rewards = 0
    # Training
    while not done:
        if render: env.render()
        action = agent.action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        rewards += reward
    return rewards



if __name__=='__main__':
    # Setup
    epsilon = 1
    disabled = not RENDER
    env = LearningEnvironment(num_particles=1, disable_render=disabled)
    agent = DDPG(env)
    agent.restore_model(PATH)

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
        test_reward = np.mean([test(env, agent) for i in range(10)])
        print("Train Reward {0}, Test Reward {1}".format(train_reward, test_reward))

        # Render
        test(env, agent, render=RENDER)

        # Save model
        agent.save_model(PATH,episode)

        # Update parameters
        epsilon *= 0.995





