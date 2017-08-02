import sys
import math
import time
import random
import pygame
import pygame.gfxdraw
import numpy as np
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn import preprocessing
from skimage.draw import circle
from simulation.utils import cart2pol
from simulation.universe import Universe
import seaborn as sns; sns.set()



class ObservationSpace:

    def __init__(self, state_size):
        self.shape = (state_size,)


class ActionSpace:

    def __init__(self, num_particles, action_dimensions):
        self.shape = (action_dimensions,)



class LearningEnvironment:
    """
    Environment that an algorithm can play with
    """
    action_dimensions = 2
    state_dimensions = 6 # The number of state dimensions (x,y,angle,speed,tx,ty)
    max_steps = 100

    screen = None
    screen_width = 800
    screen_height = 800


    def __init__(self, num_particles=4, particle_size=30, disable_render=False):
        """
        @history: A vector where each row is a previous state
        """
        self.current_step = 0
        self.num_particles = num_particles
        self.observation_space = ObservationSpace(num_particles * self.state_dimensions)
        self.action_space = ActionSpace(num_particles, self.action_dimensions)

        self.universe = Universe((self.screen_width, self.screen_height))
        self.universe.addFunctions(['move', 'bounce', 'brownian', 'collide', 'drag'])
        self.universe.mass_of_air = 0.002

        # Add all the particles
        colors = sns.color_palette("muted")
        for i in range(self.num_particles):
            name = "primary" if i==0 else "default"
            color = tuple(255*c for c in colors[i])
            target = self.universe.addTarget(radius=particle_size, color=color)
            particle = self.universe.addParticle(radius=particle_size, mass=100, speed=0, elasticity=1, color=color, target=target, name=name)

        # Add the primary particle
        self.primary = self.universe.particles[0]

        if not disable_render:
            print('Initializing pygame screen')
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Bouncing Objects')


    def step(self, action):
        """
        Step the environment forward
        Return (observation, reward, done, info)
        """
        # Particle 1 is being controlled
        angle, accel = cart2pol(-action[1],action[0])
        self.primary.accelerate((angle, accel))

        # Step forward one timestep
        collisions = self.universe.update()
        state = self.get_current_state()
        self.current_step += 1

        if self.primary.atTarget(threshold=50):
            reward = 1
            done = True
        else:
            reward = 0
            done = self.current_step >= self.max_steps or collisions > 0

        info = {'step': self.current_step}
        return state, reward, done, info


    def reset(self):
        """Respawn the particles and the targets"""
        for particle in self.universe.particles:
            particle.respawn(self.screen_width, self.screen_height)

        for target in self.universe.targets:
            target.respawn(self.screen_width, self.screen_height)

        self.current_step = 0

        return self.get_current_state()


    def render(self):
        """
        Render the environment
        """
        # Clear the screen
        self.screen.fill(self.universe.colour)

        for p in self.universe.particles:
            self.draw_circle(int(p.x), int(p.y), p.size, p.colour, filled=True)

        for t in self.universe.targets:
            self.draw_circle(int(t.x), int(t.y), t.radius, t.color, filled=False)
            self.draw_circle(int(t.x), int(t.y), int(t.radius/4), t.color, filled=True)

        # Draw the primary particle direction
        p = self.primary
        dx, dy = p.getSpeedVector()
        pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+10*dx), int(p.y+10*dy), (0,0,0))
        self.flip_screen()
        time.sleep(0.01)


    def flip_screen(self):
        """
        Flip the pygame screen and catch events
        """
        # Push to the screen
        pygame.display.flip()

        # Make sure we catch quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()


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


    def draw_circle(self, x, y, r, color, filled=False):
        """Draw circle on the screen"""
        if filled:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), r, color)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), r, color)


    def get_default_action(self):
        """
        Return a chase goal action
        """
        dx = (self.primary.target.x - self.primary.x) / self.screen_width
        dy = (self.primary.target.y - self.primary.y) / self.screen_height

        return np.array([dx,dy])


    def get_current_state(self):
        """
        Return a representation of the simulation state
        """
        state = []
        for i,particle in enumerate(self.universe.particles):
            x = particle.x / self.universe.width
            y = particle.y / self.universe.height
            a = particle.angle / (2 * math.pi)
            v = particle.speed / 10
            tx = particle.target.x / self.universe.width
            ty = particle.target.y / self.universe.height
            state.extend((x, y, a, v, tx, ty))
        return np.array(state)


    def close(self):
        """Clean up the env"""
        pass



class HumanLearningEnvironment(LearningEnvironment):
    """
    Overide base class to allow human control
    """

    def step(self,action):
        """ Override step to insert action"""
        action = self.control_loop()
        return super().step(action)


    def control_loop(self):
        """
        Return a user selected action
        """
        action = None

        while action is None:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        action = 0
                    if event.key == pygame.K_d:
                        action = 1
                    if event.key == pygame.K_s:
                        action = 2
                    if event.key == pygame.K_a:
                        action = 3
        return action


if __name__=="__main__":
    # Demo the environment
    env = HumanLearningEnvironment(num_particles=4, render=True)
    rewards = 0
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rewards += reward
        if done:
            print("Simulation complete. Reward: ",reward)
    env.render()




