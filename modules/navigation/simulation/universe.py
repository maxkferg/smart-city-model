import numpy as np
import math, random
import matplotlib
from .targets import Target
from .particles import Particle
from .utils import addVectors, normalizeAngle



class Universe:
    """ Defines the boundary of a simulation and its properties """

    def __init__(self, size):
        (width, height) = size
        self.width = width
        self.height = height
        self.targets = []
        self.particles = []
        self.springs = []

        self.colour = (255,255,255)
        self.mass_of_air = 0.2
        self.elasticity = 0.75
        self.acceleration = (0,0)

        self.particle_functions1 = []
        self.particle_functions2 = []
        self.function_dict = {
            'move': (1, lambda p: p.move()),
            'drag': (1, lambda p: p.experienceDrag()),
            'enter': (1, lambda p: self.enter(p)),
            'exit': (1, lambda p: self.exit(p)),
            'bounce': (1, lambda p: self.bounce(p)),
            'brownian': (1, lambda p: p.brownian()),
            'accelerate': (1, lambda p: p.accelerate(self.acceleration)),
            'collide': (2, lambda p1, p2: self.collide(p1, p2)),
            'combine': (2, lambda p1, p2: combine(p1, p2)),
            'attract': (2, lambda p1, p2: p1.attract(p2))
        }

    def addFunctions(self, function_list):
        for func in function_list:
            (n, f) = self.function_dict.get(func, (-1, None))
            if n == 1:
                self.particle_functions1.append(f)
            elif n == 2:
                self.particle_functions2.append(f)
            else:
                print("No such function: %s" % f)


    def addParticle(self, **kargs):
        """ Add n particles with properties given by keyword arguments """
        name = kargs.get('name', 'default')
        size = kargs.get('radius', random.randint(10, 20))
        mass = kargs.get('mass', random.randint(100, 10000))
        target = kargs.get('target', None)
        x = kargs.get('x', random.uniform(size, self.width - size))
        y = kargs.get('y', random.uniform(size, self.height - size))

        particle = Particle((x, y), size, mass, target, name=name)
        particle.speed = kargs.get('speed', 100*random.random())
        particle.angle = kargs.get('angle', random.uniform(0, math.pi*2))
        particle.colour = kargs.get('color', (0,0,0))
        particle.elasticity = kargs.get('elasticity', self.elasticity)
        particle.drag = (particle.mass/(particle.mass + self.mass_of_air)) ** particle.size

        self.particles.append(particle)


    def addTarget(self, radius, color):
        """Add a target for the particles to reach"""
        target = Target((0,0), radius, color)
        target.respawn(self.width, self.height)
        self.targets.append(target)
        return target


    def update(self):
        """
        Moves particles and tests for collisions with the walls and each other
        Return the number of particle-particle collisions
        """

        self.collisions = 0 # The number of collisions

        for i, particle in enumerate(self.particles, 1):
            for f in self.particle_functions1:
                f(particle)
            for particle2 in self.particles[i:]:
                for f in self.particle_functions2:
                    f(particle, particle2)
            # Fix all the angles
            particle.angle = normalizeAngle(particle.angle)

        for spring in self.springs:
            spring.update()

        return self.collisions


    def bounce(self, particle):
        """ Tests whether a particle has hit the boundary of the environment """

        if particle.x > self.width - particle.size:
            particle.x = 2*(self.width - particle.size) - particle.x
            particle.angle = - particle.angle
            particle.speed *= self.elasticity

        elif particle.x < particle.size:
            particle.x = 2*particle.size - particle.x
            particle.angle = - particle.angle
            particle.speed *= self.elasticity

        if particle.y > self.height - particle.size:
            particle.y = 2*(self.height - particle.size) - particle.y
            particle.angle = math.pi - particle.angle
            particle.speed *= self.elasticity

        elif particle.y < particle.size:
            particle.y = 2*particle.size - particle.y
            particle.angle = math.pi - particle.angle
            particle.speed *= self.elasticity


    def collide(self, p1, p2):
        """ Tests whether two particles overlap
            If they do, make them bounce, i.e. update their angle, speed and position """

        dx = p1.x - p2.x
        dy = p1.y - p2.y

        dist = math.hypot(dx, dy)
        if dist < p1.size + p2.size:
            angle = math.atan2(dy, dx) + 0.5 * math.pi
            total_mass = p1.mass + p2.mass

            (p1.angle, p1.speed) = addVectors((p1.angle, p1.speed*(p1.mass-p2.mass)/total_mass), (angle, 2*p2.speed*p2.mass/total_mass))
            (p2.angle, p2.speed) = addVectors((p2.angle, p2.speed*(p2.mass-p1.mass)/total_mass), (angle+math.pi, 2*p1.speed*p1.mass/total_mass))
            elasticity = p1.elasticity * p2.elasticity
            p1.speed *= elasticity
            p2.speed *= elasticity

            overlap = 0.5*(p1.size + p2.size - dist+1)
            p1.x += math.sin(angle)*overlap
            p1.y -= math.cos(angle)*overlap
            p2.x -= math.sin(angle)*overlap
            p2.y += math.cos(angle)*overlap

            if "primary" in [p1.name, p2.name]:
                self.collisions += 1


    def findParticle(self, x, y):
        """ Returns any particle that occupies position x, y """

        for particle in self.particles:
            if math.hypot(particle.x - x, particle.y - y) <= particle.size:
                return particle
        return None
