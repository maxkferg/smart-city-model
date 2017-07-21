import time
import random
import pygame
import pygame.gfxdraw
import numpy as np
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from simulation import particles
from skimage.draw import circle



class ObservationSpace:

    def __init__(self, history_length, state_size):
        self.shape = (history_length, state_size)


class ActionSpace:

    def __init__(self, num_particles, num_actions):
        self.n = num_actions
        self.shape = (1,)

    def sample(self):
        return random.randint(0, 2)
        #return np.random.uniform(low=-1.0, high=1.0, size=self.shape)


class LearningEnvironment:
    """
    Environment that an algorithm can play with
    """
    dimensions = 3 # The number of state dimensions (x,y,angle)
    max_steps = 1000
    max_particles = 4 # The maximum number of objects we can track
    num_particles = 4 # The actual number of particles
    num_actions = 3
    max_rotation = 0.6 # The maximum rotation per time step
    history_length = 10 # The number of steps to save

    screen = None
    screen_width = 800
    screen_height = 800

    def __init__(self, skip=4):
        """
        @history: A vector where each row is a previous state
        """
        state_size = self.max_particles * self.dimensions

        self.skip = skip
        self.current_step = 0
        self.observation_space = ObservationSpace(self.history_length, state_size)
        self.action_space = ActionSpace(self.num_particles, self.num_actions)

        self.history = np.zeros(self.observation_space.shape)
        self.universe = particles.Environment((self.screen_width, self.screen_height))
        self.universe.colour = (255,255,255)
        self.universe.addFunctions(['move', 'bounce', 'brownian', 'collide', 'drag'])
        self.universe.mass_of_air = 0.001

        for p in range(self.num_particles):
            self.universe.addParticles(mass=100, size=50, speed=20, elasticity=1)

        # Populate the history
        self.reset()


    def step(self,action):
        """
        Step the environment forward
        Return (observation, reward, done, info)
        """
        # Skip forward a few frames
        rewards = 0
        for i in range(self.skip):
            state, reward, done, info = self._step(action)
            rewards += reward
            if done:
                break
        return state, reward, done, info


    def _step(self,action):
        """
        Step the environment forward
        Return (observation, reward, done, info)
        """
        # Particle 1 can't turn as much
        if action==0:
            self.universe.particles[0].rotate(-0.6)
        if action==1:
            self.universe.particles[0].rotate(0.0)
        if action==2:
            self.universe.particles[0].rotate(0.6)

        collisions = self.universe.update()
        state = self.get_current_state()
        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1,:] = state
        self.current_step += 1

        reward = -collisions - np.sum(abs(action))/self.max_steps
        done = self.current_step >= self.max_steps
        info = {'step': self.current_step}
        return (self.history, reward, done, info)


    def reset(self):
        """ """
        # Step the environment a few times to move the particles
        for particle in self.universe.particles[1:]:
            particle.respawn(self.screen_width, self.screen_height)

        self.universe.particles[0].name = "primary"
        self.universe.particles[0].colour = (0,255,0)

        for i in range(self.history_length):
            action = self.action_space.sample()
            self.step(action)

        self.current_step = 0
        return self.history


    def render(self):
        """
        Render the environment
        """
        if not self.screen:
            print('Initializing pygame screen')
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Bouncing Objects')

        # Clear the screen
        self.screen.fill(self.universe.colour)

        for p in self.universe.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), p.size, p.colour)
            pygame.gfxdraw.aacircle(     self.screen, int(p.x), int(p.y), p.size, p.colour)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        pygame.display.flip()
        time.sleep(0.001)


    def draw(self,scale=10):
        """
        Draw the current state on a black and white image
        """
        width = int(self.screen_width/scale)
        height = int(self.screen_height/scale)
        img = np.zeros((height,width),dtype=np.uint8)
        for p in self.universe.particles:
            rr, cc = circle(p.x/scale, p.y/scale, p.size/scale)
            rr[rr>79] = 79; rr[rr<0] = 0
            cc[cc>79] = 79; cc[cc<0] = 0
            img[rr, cc] = 1
        return img


    def get_current_state(self):
        """
        Return a representation of the simulation state
        """
        state = np.zeros((self.max_particles, 3))
        for i,particle in enumerate(self.universe.particles):
            x = particle.x / self.universe.width
            y = particle.y / self.universe.height
            a = particle.angle
            state[i,:] = (x, y, a)
        return state.flatten()


    def close(self):
        """Clean up the env"""
        pass