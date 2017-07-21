import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ddpg import DDPG
from tqdm import tqdm
from ddpg.space_network import SpaceNetwork
from environment import LearningEnvironment


PATH = 'models'
EPISODES = 10000
BATCH_SIZE = 128
STEPS = 4


with tf.Session() as sess:
    # Setup
    env = LearningEnvironment()
    space = SpaceNetwork(
        sess=sess,
        n_input=env.observation_space.shape[1],
        n_steps=STEPS,
        n_hidden=100,
        output_size=(80,80)
    )

    # Initialize the graph
    sess.run(tf.global_variables_initializer())

    # Train
    for episode in range(EPISODES):
        done = False
        state = env.reset()
        rewardEpisode = []
        shouldRender = (episode%10==0)

        inputs = []
        outputs = []

        for i in range(BATCH_SIZE):
            next_state, reward, done, info = env.step(0)
            if shouldRender:
                pass
                #env.render()

            state = next_state[0:STEPS, :]
            output = env.draw()

            inputs.append(state)
            outputs.append(output)

        inputs = np.stack(inputs)
        outputs = np.stack(outputs)

        space.train_on_batch(inputs, outputs)

        if episode%10==0:
            loss = space.test_on_batch(inputs, outputs)
            print('loss: ',loss)


