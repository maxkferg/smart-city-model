from math import pi
import random
import pygame
import particles

(width, height) = (400, 400)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Springs')

universe = particles.Environment((width, height))
universe.colour = (255,255,255)
universe.addFunctions(['move','enter', 'exit', 'brownian', 'collide', 'drag'])
universe.mass_of_air = 0.02

for p in range(4):
    universe.addParticles(mass=100, size=16, speed=2, elasticity=1, colour=(20,40,200))

selected_particle = None
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    universe.update()
        
    screen.fill(universe.colour)
    
    for p in universe.particles:
        pygame.draw.circle(screen, p.colour, (int(p.x), int(p.y)), p.size, 0)
        
    for s in universe.springs:
        pygame.draw.aaline(screen, (0,0,0), (int(s.p1.x), int(s.p1.y)), (int(s.p2.x), int(s.p2.y)))

    pygame.display.flip()