import sys
import pygame
from pygame.locals import *
from pygame.color import THECOLORS
from pygame.draw import polygon 
from intersection import intersection


class Renderer():
    """
    A pygame renderer for the intersection model
    """

    def __init__(self,model):
        """Create the new renderer"""
        pygame.init()
        self.model = model
        self.screen = pygame.display.set_mode((model.width, model.height))
        self.clock = pygame.time.Clock()
        self.screen.set_alpha(None)
        print("Initialized PyGame")


    def draw(self):
        """Draw the intersection model with pygame"""
        for shape in self.model.shapes:
            shape.draw(self.screen)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()

 




renderer = Renderer(intersection)
while True:
    renderer.draw()