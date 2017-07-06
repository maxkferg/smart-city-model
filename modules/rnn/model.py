import time
import pygame
import numpy as np
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from simulation import particles

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=120)


width = 800
height = 800

hidden_layers = 100
max_objects = 5 # The maximum number of objects we can track

input_shape = 2*max_objects # (x,y) for five objects
input_sequence_len = 12 # Use the previous 12 frames
output_shape = input_shape # A single output vector
output_sequence_len = 1 # Only predict the next value
batch_size = 100*128 # Train with batches of 128

model = Sequential()
model.add(LSTM(
    units=hidden_layers,
    input_shape=(input_sequence_len,input_shape),
))
model.add(Dense(output_shape))
model.compile('adam', 'mse')  # Or categorical_crossentropy

########### TEST ###########


def get_state(environment):
    """
    Return a representation of the simulation state
    """
    dimensions = 2
    state = np.zeros((max_objects,dimensions))
    for i,particle in enumerate(environment.particles):
        x = particle.x/environment.width
        y = particle.y/environment.height
        state[i,:] = (x, y)
    return state.flatten()



if __name__=='__main__':

    while True:

        # Create environment
        universe = particles.Environment((width, height))
        universe.colour = (255,255,255)
        universe.addFunctions(['move', 'bounce', 'brownian', 'collide', 'drag'])
        universe.mass_of_air = 0.02

        for p in range(4):
            universe.addParticles(mass=100, size=16, speed=2, elasticity=1)

        # Run the simulation for a few steps to build a history
        history = np.zeros((input_sequence_len, input_shape))
        for i in range(input_sequence_len):
            history[i,:] = get_state(universe)
            universe.update()

        # Now we get a new point every time we update
        batch = np.zeros((batch_size, input_sequence_len, input_shape))
        labels = np.zeros((batch_size, input_shape))
        for i in range(batch_size):
            history = np.roll(history, shift=-1, axis=0)
            history[-1,:] = get_state(universe)

            batch[i,:,:] = history
            # Update the universe and save the next state as the label
            universe.update()
            labels[i,:] = get_state(universe)

        model.fit(batch, labels, epochs=1)
        model.test_on_batch(batch, labels)




        # Simulate the results
        history = np.zeros((input_sequence_len, input_shape))
        screen = pygame.display.set_mode((universe.width, universe.height))
        pygame.display.set_caption('Springs')

        # Just simulate 100 frames
        last_predictions = np.zeros((max_objects,2))
        for i in range(100):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            universe.update()
                
            screen.fill(universe.colour)

            history = np.roll(history, shift=-1, axis=0)
            history[-1,:] = get_state(universe)
            batch = history[None,:,:]
            predictions = model.predict(batch).reshape((-1,2))
            
            for p in universe.particles:
                pygame.draw.circle(screen, p.colour, (int(p.x), int(p.y)), p.size, 0)
                
            time.sleep(0.1)

            # Plot the predictions
            for row in range(max_objects):
                if row<len(universe.particles):
                    particle = universe.particles[row]
                x = int(universe.width * predictions[row,0])
                y = int(universe.height * predictions[row,1])
                pygame.draw.circle(screen, particle.colour, (x,y), int(1.5*p.size), 2)

            # Compare the prediction to the actual event
            for row in range(max_objects):
                if row<len(universe.particles):
                    particle = universe.particles[row]
                x = int(universe.width * last_predictions[row,0])
                y = int(universe.height * last_predictions[row,1])
                pygame.draw.circle(screen, (0,0,0), (x,y), int(0.2*p.size), 0)

            last_predictions = predictions

            pygame.display.flip()


