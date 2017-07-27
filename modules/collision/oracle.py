import time
import random
import numpy as np
import tensorflow as tf
from ddpg import DDPG
from tqdm import tqdm
from dqn.oracle import OracleNetwork
from environment import LearningEnvironment


PATH = 'models'
EPISODES = 1000*1000
BUFFER_LENGTH = 1000


class StateBuffer():

    def __init__(self, batch_size, buffer_length, state_size):
        self.total_steps = pre_steps+post_steps
        self.buffer_length = buffer_length
        self.states = np.zeros((batch_size, buffer_length, state_size))
        self.rewards = np.zeros((batch_size, buffer_length))

    def add(self,states,rewards):
        """Add a batch of rewards/states to the buffer"""
        self.rewards = np.roll(self.rewards, shift=-1, axis=1)
        self.states = np.roll(self.states, shift=-1, axis=1)
        self.states[:,-1,:] = states
        self.rewards[:,-1] = rewards

    def get_random_sample(self, n_timesteps):
        """
        Return a random (state,reward) sample
        The sample size will be [batch_size, n_timesteps, state_size]
        """
        start_index = random.randint(0, self.buffer_length-n_timesteps)
        last_index = start_index + n_timesteps
        states = self.states[:, start_index:last_index, :]
        rewards = self.rewards[:, start_index:last_index, None]
        return states, rewards



class EnvironmentBuffer(StateBuffer):

    def __init__(self, batch, *args, **kwargs):
        super().__init__(batch, *args, **kwargs)
        self.envs = [LearningEnvironment() for _ in range(batch)]

    def step(self):
        actions = [e.action_space.sample() for e in self.envs]
        results = [env.step(a) for env,a in zip(self.envs, actions)]
        states = [result[0] for result in results]
        rewards = [result[1] for result in results]
        self.add(states, rewards)
        #self.envs[0].render()



def draw(buff, sources_batch, targets_batch, outputs_batch):
    """Render a test attempt"""

    ENV = 0

    # Setup the screen
    environment = buff.envs[ENV]
    environment.init_screen()
    environment.screen.fill(environment.universe.colour)

    # Print the initial positions
    for timestep in range(sources_batch.shape[1]):
        for particle in range(0, sources_batch.shape[2], 4):
            x = sources_batch[ENV, timestep, particle]
            y = sources_batch[ENV, timestep, particle+1]
            x *= environment.screen_width
            y *= environment.screen_height
            environment.draw_circle(x, y, 5, (0,0,10*particle), filled=True)
        time.sleep(0.02)

    # Print the target positions (actual and predicted)
    for timestep in range(targets_batch.shape[1]):
        for particle in range(0, targets_batch.shape[2], 4):
            # Draw the actual target positions
            x = targets_batch[ENV, timestep, particle]
            y = targets_batch[ENV, timestep, particle+1]
            x *= environment.screen_width
            y *= environment.screen_height
            environment.draw_circle(x, y, 5, (0,0,10*particle))
            # Draw the predicted target positions
            x = outputs_batch[ENV, timestep, particle]
            y = outputs_batch[ENV, timestep, particle+1]
            x *= environment.screen_width
            y *= environment.screen_height
            environment.draw_circle(x, y, 5, (255,0,10*particle))
        # Draw on the screen
        environment.flip_screen()
        time.sleep(0.05)




# Setup
batch_size = 64
pre_steps = 3  # Use the most recent three steps
post_steps = 5 # Predict 5 steps into the future
total_steps = pre_steps + post_steps
learning_rate = 0.01
save_directory = 'results/collision'
save_name = 'oracle'


if __name__=='__main__':

    state_size = LearningEnvironment().observation_space.shape[0]

    model = OracleNetwork(
        rnn_size=256,
        num_layers=3,
        num_features=state_size
    )

    init = tf.global_variables_initializer()
    buff = EnvironmentBuffer(batch_size, BUFFER_LENGTH, state_size)

    print("Populating buffer")
    for timestep in tqdm(range(BUFFER_LENGTH)):
        buff.step()

    # Train
    with tf.Session() as sess:
        sess.run(init)
        model.restore(sess, save_directory, save_name)

        for episode in range(EPISODES):
            # Collect some new samples
            state_batch, rewards_batch = buff.get_random_sample(total_steps)
            sources_batch = state_batch[:,:pre_steps,:]
            targets_batch = state_batch[:,pre_steps:,:]
            rewards_batch = rewards_batch[:,pre_steps:,:]
            sources_length = batch_size*[sources_batch.shape[1]]
            targets_length = batch_size*[targets_batch.shape[1]]

            # Add a new sample
            buff.step()

            # Train on the buffer
            model.train(
                sess=sess,
                sources_batch=sources_batch,
                targets_batch=targets_batch,
                learning_rate=learning_rate,
                sources_lengths=sources_length,
                targets_lengths=targets_length
            )

            if episode%100==0:
                # Evaluate on this dataset
                tloss, eloss = model.evaluate(
                    sess=sess,
                    sources_batch=sources_batch,
                    targets_batch=targets_batch,
                    learning_rate=learning_rate,
                    sources_lengths=sources_length,
                    targets_lengths=targets_length
                )
                print('Episode {0}, Train Loss {1:.4f}, Eval Loss {2:.4f}'.format(episode,tloss,eloss))

            if episode%100==0:
                outputs_batch = model.predict(
                    sess=sess,
                    sources_batch=sources_batch,
                    sources_lengths=sources_length,
                    targets_lengths=targets_length
                )
                draw(buff, sources_batch, targets_batch, outputs_batch)

            if episode and episode%5000==0:
                learning_rate = 0.95*learning_rate + 0.0001
                print("New learning rate: %.5f"%learning_rate)
                model.save(sess, save_directory, save_name, episode)






