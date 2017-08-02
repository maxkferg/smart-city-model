import numpy as np
import math, random
import matplotlib
from .utils import addVectors, pol2cart


def combine(p1, p2):
    if math.hypot(p1.x - p2.x, p1.y - p2.y) < p1.size + p2.size:
        total_mass = p1.mass + p2.mass
        p1.x = (p1.x*p1.mass + p2.x*p2.mass)/total_mass
        p1.y = (p1.y*p1.mass + p2.y*p2.mass)/total_mass
        (p1.angle, p1.speed) = addVectors((p1.angle, p1.speed*p1.mass/total_mass), (p2.angle, p2.speed*p2.mass/total_mass))
        p1.speed *= (p1.elasticity*p2.elasticity)
        p1.mass += p2.mass
        p1.collide_with = p2



class Particle:
    """ A circular object with a velocity, size and mass """

    def __init__(self, position, size, mass=1, target=None, name="default"):
        (x, y) = position
        self.x = x
        self.y = y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 0
        self.speed = 0
        self.angle = 0
        self.mass = mass
        self.drag = 1 # Things slow down without drag
        self.elasticity = 0.9
        self.target = target
        self.noise = np.array([0,0])
        self.name = name


    def move(self):
        """ Update position based on speed, angle """
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed

    def atTarget(self, threshold=10):
        """Return True if this particle is close enough to its target"""
        dx = abs(self.x - self.target.x)
        dy = abs(self.y - self.target.y)
        return (dx**2 + dy**2 < threshold**2)


    def getSpeedVector(self):
        """Return the speed vector in cartesion coordinates"""
        dx, dy = pol2cart(self.angle, self.speed)
        return dx, dy

    def experienceDrag(self):
        self.speed *= self.drag


    def accelerate(self, vector):
        """ Change angle and speed by a given vector """
        (self.angle, self.speed) = addVectors((self.angle, self.speed), vector)


    def rotate(self, angle):
        """ Rotate the particle by a certain angle"""
        self.angle += angle


    def brownian(self):
        """Add some correlated acceleration"""
        #change = np.random.uniform(-1,1,size=2)
        #vector = 0.5*self.noise + 1.0*change + 0.6*self.speed
        #vector = pol2cart(self.angle, 0.003*self.speed) # Antidrag
        #polar = cart2pol(vector[0],vector[1])
        #self.accelerate(polar)
        #self.noise = vector / np.linalg.norm(vector)
        if self.name!="primary":
            self.speed = 10

    def attract(self, other):
        """" Change velocity based on gravatational attraction between two particle"""

        dx = (self.x - other.x)
        dy = (self.y - other.y)
        dist  = math.hypot(dx, dy)

        if dist < self.size + other.size:
            return True

        theta = math.atan2(dy, dx)
        force = 0.1 * self.mass * other.mass / dist**2
        self.accelerate((theta - 0.5 * math.pi, force/self.mass))
        other.accelerate((theta + 0.5 * math.pi, force/other.mass))

    def respawn(self,width,height):
        """Respawn this particle somewhere else"""
        x = random.uniform(0,width)
        y = random.uniform(0,height)
        if random.random() < 0.5:
            x = random.choice([0,width])
        else:
            y = random.choice([0,height])
        self.x = x
        self.y = y

