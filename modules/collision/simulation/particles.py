import numpy as np
import math, random
import matplotlib
from pylab import get_cmap
from matplotlib.colors import ColorConverter



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def addVectors(p1, p2):
    """ Returns the sum of two vectors """
    (angle1, length1) = p1
    (angle2, length2) = p2

    x  = math.sin(angle1) * length1 + math.sin(angle2) * length2
    y  = math.cos(angle1) * length1 + math.cos(angle2) * length2

    angle  = 0.5 * math.pi - math.atan2(y, x)
    length = math.hypot(x, y)

    return (angle, length)


def combine(p1, p2):
    if math.hypot(p1.x - p2.x, p1.y - p2.y) < p1.size + p2.size:
        total_mass = p1.mass + p2.mass
        p1.x = (p1.x*p1.mass + p2.x*p2.mass)/total_mass
        p1.y = (p1.y*p1.mass + p2.y*p2.mass)/total_mass
        (p1.angle, p1.speed) = addVectors((p1.angle, p1.speed*p1.mass/total_mass), (p2.angle, p2.speed*p2.mass/total_mass))
        p1.speed *= (p1.elasticity*p2.elasticity)
        p1.mass += p2.mass
        p1.collide_with = p2


def getColor():
    """Return a random color from Matplotlib"""
    converter = ColorConverter()
    colors = get_cmap("Set1").colors
    color = random.choice(colors)
    return (255*np.array(color)).astype(int)


class Particle:
    """ A circular object with a velocity, size and mass """

    def __init__(self, position, size, mass=1, name="default"):
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
        self.noise = np.array([0,0])
        self.name = name


    def move(self):
        """ Update position based on speed, angle """
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed


    def experienceDrag(self):
        self.speed *= self.drag


    def accelerate(self, vector):
        """ Change angle and speed by a given vector """
        (self.angle, self.speed) = addVectors((self.angle, self.speed), vector)


    def rotate(self, angle):
        """ Rotate the particle by a certain angle """
        self.angle += angle


    def brownian(self):
        """Add some correlated acceleration"""
        #change = np.random.uniform(-1,1,size=2)
        #vector = 0.5*self.noise + 1.0*change + 0.6*self.speed
        #vector = pol2cart(self.angle, 0.003*self.speed) # Antidrag
        #polar = cart2pol(vector[0],vector[1])
        #self.accelerate(polar)
        #self.noise = vector / np.linalg.norm(vector)
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
        self.colour = getColor()


class Environment:
    """ Defines the boundary of a simulation and its properties """

    def __init__(self, size):
        (width, height) = size
        self.width = width
        self.height = height
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

    def addParticles(self, n=1, **kargs):
        """ Add n particles with properties given by keyword arguments """

        for i in range(n):
            size = kargs.get('size', random.randint(10, 20))
            mass = kargs.get('mass', random.randint(100, 10000))
            x = kargs.get('x', random.uniform(size, self.width - size))
            y = kargs.get('y', random.uniform(size, self.height - size))

            particle = Particle((x, y), size, mass)
            particle.speed = kargs.get('speed', 100*random.random())
            particle.angle = kargs.get('angle', random.uniform(0, math.pi*2))
            particle.colour = kargs.get('colour', getColor())
            particle.elasticity = kargs.get('elasticity', self.elasticity)
            particle.drag = (particle.mass/(particle.mass + self.mass_of_air)) ** particle.size

            self.particles.append(particle)

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

        for spring in self.springs:
            spring.update()

        return self.collisions


    def enter(self, particle):
        """Add new particles if there are less than 3"""
        if len(self.particles)>=3:
            return
        x = random.uniform(0,self.width)
        y = random.uniform(0,self.height)
        if random.random() < 0.5:
            x = random.choice([0,self.width])
        else:
            y = random.choice([0,self.height])
        self.addParticles(x=x,y=y)


    def exit(self, particle):
        """ Tests whether a particle has left the environment """
        if particle.x > self.width + particle.size:
            particle.respawn(self.width, self.height)

        elif particle.x < -particle.size:
            particle.respawn(self.width, self.height)

        elif particle.y > self.height - particle.size:
            particle.respawn(self.width, self.height)

        elif particle.y < -particle.size:
            particle.respawn(self.width, self.height)


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
