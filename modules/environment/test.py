import sys
import pygame
from pygame.locals import *
from pygame.color import THECOLORS
from pygame.draw import polygon
from .models import auburn
from .renderers.simple import Renderer



def run():
    """Run the test"""
    renderer = Renderer(auburn.intersection)
    while True:
        renderer.draw()