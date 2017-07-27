import time
import random
import threading
import numpy as np
import tensorflow as tf
from ddpg import DDPG
from tqdm import tqdm
from dqn.oracle import OracleNetwork
from tfmodels.buffers import HistoryBuffer
from environment import LearningEnvironment




def draw(environment, sources_batch, targets_batch, outputs_batch):
    """Render a test attempt"""

    # Setup the screen
    sample_id = 0
    environment.init_screen()
    environment.screen.fill(environment.universe.colour)

    # Print the initial positions
    for timestep in range(sources_batch.shape[1]):
        for particle in range(0, sources_batch.shape[2], 4):
            x = sources_batch[sample_id, timestep, particle]
            y = sources_batch[sample_id, timestep, particle+1]
            x *= environment.screen_width
            y *= environment.screen_height
            environment.draw_circle(x, y, 5, (0,0,10*particle), filled=True)
        time.sleep(0.02)

    # Print the target positions (actual and predicted)
    for timestep in range(targets_batch.shape[1]):
        for particle in range(0, targets_batch.shape[2], 4):
            # Draw the actual target positions
            x = targets_batch[sample_id, timestep, particle]
            y = targets_batch[sample_id, timestep, particle+1]
            x *= environment.screen_width
            y *= environment.screen_height
            environment.draw_circle(x, y, 5, (0,0,10*particle))
            # Draw the predicted target positions
            x = outputs_batch[sample_id, timestep, particle]
            y = outputs_batch[sample_id, timestep, particle+1]
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

total_epochs = 1000
epoch_length = 50
enqueue_threads = 4



# Define the environment and history buffer
environment = LearningEnvironment()
state_size = environment.observation_space.shape[0]

# define shapes
sources_shape = [pre_steps, state_size]
targets_shape = [post_steps, state_size]

# Define the palceholder we use for feeding data into the queue
sources_placeholder = tf.placeholder(tf.float32, sources_shape, name="sources_placeholder")
targets_placeholder = tf.placeholder(tf.float32, targets_shape, name="targets_placeholder")
sources_length_placeholder = tf.placeholder(tf.int32, [], name="sources_length_placeholder")
targets_length_placeholder = tf.placeholder(tf.int32, [], name="targets_length_placeholder")


experience = tf.RandomShuffleQueue(capacity=200*batch_size,
                                   min_after_dequeue=100*batch_size,
                                   dtypes=[tf.float32, tf.float32, tf.int32, tf.int32],
                                   shapes=[sources_shape, targets_shape, [], []],
                                   names=["sources", "targets", "sources_length", "targets_length"],
                                   name='experience_replay')

# Define the queue ops
enqueue_op = experience.enqueue({
    "sources": sources_placeholder,
    "targets": targets_placeholder,
    "sources_length": sources_length_placeholder,
    "targets_length": targets_length_placeholder
})

deque_single_op = experience.dequeue()
deque_batch_op = experience.dequeue_many(batch_size)

# Define the tensorflow model
print("Building Oracle Network")
model = OracleNetwork(
    rnn_size=256,
    num_layers=3,
    num_features=state_size,
    sources_batch=deque_batch_op["sources"],
    targets_batch=deque_batch_op["targets"],
    sources_lengths=deque_batch_op["sources_length"],
    targets_lengths=deque_batch_op["targets_length"]
)

coord = tf.train.Coordinator()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    model.restore(sess, save_directory, save_name)

    # Define a custom thread for running the enqueue op that grabs a new
    # screen in a loop and feeds it to the placeholder.
    # Each thread needs to fetch samples from its own environment to prevent race conditions
    def enqueue_thread():
        environment = LearningEnvironment()
        environment_buffer = HistoryBuffer(environment, total_steps, state_size)
        with coord.stop_on_exception():
            while not coord.should_stop():
                state, reward = environment_buffer.fetch()
                # Feed in all the values the queue
                sess.run(enqueue_op, feed_dict={
                    sources_placeholder: state[:pre_steps, :],
                    targets_placeholder: state[pre_steps:, :],
                    sources_length_placeholder: pre_steps,
                    targets_length_placeholder: post_steps
                })

    # Start queuing new samples
    for i in range(enqueue_threads):
        print("Starting enqueue operation %i"%i)
        threading.Thread(target=enqueue_thread).start()

    # Run the main training loop
    for epoch in range(total_epochs):

        # Train for an epoch
        print("Training:")
        progress = tqdm(range(epoch_length))
        for _ in progress:
            loss = model.train(sess=sess, learning_rate=learning_rate)
            progress.set_postfix(loss=loss)

        # Evaluate on this dataset
        tloss, eloss = model.evaluate(sess=sess)
        print('Evaluating Epoch {0}'.format(epoch))
        print('Train Loss {1:.4f}, Eval Loss {2:.4f}'.format(epoch,tloss,eloss))

        # Draw the result
        sources_batch, targets_batch, outputs_batch = model.predict(sess=sess)
        draw(environment, sources_batch, targets_batch, outputs_batch)

        # Save to a checkpoint
        learning_rate = 0.95*learning_rate + 0.0001
        print("New learning rate: %.5f"%learning_rate)

        if epoch%10==0:
            model.save(sess, save_directory, save_name, epoch)

