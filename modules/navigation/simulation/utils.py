import numpy as np
import math


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def pol2cart(phi, rho):
    x = rho * np.sin(phi)
    y = -rho * np.cos(phi)
    return (x, y)



def normalizeAngle(angle):
    """Ensure that the angle is on a 0 to 2pi scale"""
    if angle<0:
        angle = angle + 2*math.pi
    elif angle > 2*math.pi:
        angle = angle % (2*math.pi)
    return angle


def addVectors(p1, p2):
    """ Returns the sum of two vectors """
    (angle1, length1) = p1
    (angle2, length2) = p2

    x  = math.sin(angle1) * length1 + math.sin(angle2) * length2
    y  = math.cos(angle1) * length1 + math.cos(angle2) * length2

    angle  = 0.5 * math.pi - math.atan2(y, x)
    length = math.hypot(x, y)

    return (angle, length)

