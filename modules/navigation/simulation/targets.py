import numpy as np
import math, random
import matplotlib



class Target:
    """ A circular object with a size"""

    def __init__(self, position, radius, color):
        (x, y) = position
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def respawn(self, screen_width, screen_height):
        """ Respawn the target """
        self.x = np.random.randint(low=self.radius, high=(screen_width-self.radius))
        self.y = np.random.randint(low=self.radius, high=(screen_height-self.radius))