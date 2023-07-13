import pygame

from src.utils import scale_image
from src.Cars import PlayerCar
from src.GameInfo import GameInfo
from src.Sensor import Sensor

pygame.font.init()

MAIN_FONT = pygame.font.SysFont("comicsans", 40)

BACKGROUND = scale_image(pygame.image.load("imgs/green-grass-background.jpg"), 0.35)

TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)

TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Car Driving")

FPS = 60

def draw(win, images, player_car, game_info):
    for img, pos in images:
        win.blit(img, pos)

    time_text = MAIN_FONT.render(f"Time: {game_info.get_level_time()}s", 1, (255, 255, 255))
    win.blit(time_text, (10, HEIGHT - time_text.get_height() - 40))

    vel_text = MAIN_FONT.render(f"Vel: {round(player_car.vel, 1)}px/s", 1, (255, 255, 255))
    win.blit(vel_text, (10, HEIGHT - vel_text.get_height() - 10))

    player_car.draw(win)

    #pygame.display.update()

def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False
    turn_right = False
    turn_left = False

    if keys[pygame.K_a] and player_car.vel !=0:
        player_car.rotate(left=True)
        turn_left = True
    if keys[pygame.K_d] and player_car.vel !=0:
        player_car.rotate(right=True)
        turn_right = True
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward(turn_left, turn_right)
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backwards(turn_left, turn_right)
    if keys[pygame.K_r]:
        player_car.reset()
        game_info.reset()

    if not moved:
        player_car.reduce_speed(turn_left, turn_right)


def handle_collision(player_car):
    if player_car.collide(TRACK_BORDER_MASK) != None:
        player_car.bounce()
        # Add a delay after bounce (where no input allowed)
        bounce_flag = 5
    else:
        bounce_flag = 0
    return bounce_flag

run = True
clock = pygame.time.Clock()
images = [(BACKGROUND, (0,0)), (TRACK, (0,0))]

player_car = PlayerCar(6, 5)
game_info = GameInfo()
beam_sensors = Sensor(WIN, TRACK_BORDER)

bounce_flag = 0

while run:
    clock.tick(FPS)
    draw(WIN, images, player_car, game_info)

    if bounce_flag > 0:
        bounce_flag -= 1
        player_car.reduce_speed(False, False)
    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        move_player(player_car)

        distances = beam_sensors.beam_distances(player_car)

        bounce_flag = handle_collision(player_car)

    pygame.display.update()

pygame.quit()
