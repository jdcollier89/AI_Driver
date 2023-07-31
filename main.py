import pygame

from src.Game import Game

run = True
game = Game()
FPS = 60
clock = pygame.time.Clock()


while run:
    clock.tick(FPS)
    game.draw()

    run, model_input = game.manual_loop()
    pygame.display.update()

pygame.quit()
