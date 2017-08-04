import math
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate
from skimage.transform import resize



def scale_image(image):
    """
    Scale image to 800 x 800
    """
    n = 800
    return cv.resize(image, (n, n))


def map_to_colors(pixels):
    """
    Map image from intensity to colors
    """
    pixels = -pixels # Prefe colormap reversed
    minq = np.min(pixels)
    maxq = np.max(pixels)
    pixels = 255 / (maxq-minq) * (pixels - minq)

    pixels = pixels.astype(np.float32)
    image = cv.cvtColor(pixels, cv.COLOR_GRAY2BGR)
    image = cv.convertScaleAbs(image)
    colors = cv.applyColorMap(image, cv.COLORMAP_JET)
    colors = cv.cvtColor(colors, cv.COLOR_BGR2RGB)
    return colors



def save_image_to_file(filename,pixels):
    cv.imwrite(filename, pixels)



def get_q(env, agent, action, n=40):
    """
    Return the Q value for a grid of x/y values
    Assumes all the particles are just doing what they are doing
    """
    xn = np.linspace(0, 1, n)
    yn = np.linspace(0, 1, n)
    x, y = np.meshgrid(xn, yn, indexing='ij')
    x = x.flatten() # primary position is a grid
    y = y.flatten() # Primary position is a grid

    state = env.get_current_state()
    state_batch = np.tile(state, (n**2,1))
    state_batch[:,0] = x
    state_batch[:,1] = y

    action1 = action[0] * np.ones_like(x)
    action2 = action[1] * np.ones_like(x)
    action_batch = np.stack((action1, action2)).T

    # Get Q values
    q = agent.critic_network.q_value(state_batch, action_batch)
    q = np.reshape(q, (n,n))
    return q


def get_q_background(env, agent, action):
    """
    Return a q background image
    """
    q = get_q(env, agent, action, n=400)
    q = map_to_colors(q)
    q = scale_image(q)
    return q




def plot_q_max(env, agent):
    """Plot the Q network for a range of generated states"""
    for a1 in np.linspace(-1, 1, 10):
        for a2 in np.linspace(-1, 1, 10):
            # Extract q value from the network
            q = get_q(env, agent, dx=0, dy=0, a1=0, a2=0, n=60)

            # Save this mini image to a file
            filename = "results/images/action {0:.1f} {1:.1f}.png".format(a1,a2)
            image = map_to_colors(q)
            save_image_to_file(filename, image)

    qnet = get_q(env, agent, dx=0, dy=0, a1=0, a2=0, n=800)
    return map_to_colors(qnet)



