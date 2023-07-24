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

# Change behaviour depending on whether computer or person controlling
MANUAL_CONTROL = True

player_car = PlayerCar(6, 5)
game_info = GameInfo()
beam_sensors = Sensor(WIN, TRACK_BORDER)

FPS = 60


def draw(win, images, player_car, game_info):
    """
    Draw the background, text and player car to the screen
    """
    for img, pos in images:
        win.blit(img, pos)

    time_text = MAIN_FONT.render(f"Time: {game_info.get_level_time()}s", 1, (255, 255, 255))
    win.blit(time_text, (10, HEIGHT - time_text.get_height() - 40))

    vel_text = MAIN_FONT.render(f"Vel: {round(player_car.vel, 1)}px/s", 1, (255, 255, 255))
    win.blit(vel_text, (10, HEIGHT - vel_text.get_height() - 10))

    player_car.draw(win)


def detect_input():
    """
    Detect the manual input and assign appropriate action number.
    Action number will then be picked up and used to denote what the car does (in later function).
    By breaking it out in this way, the AI input can be to decide on which 
    action to take, instead of which buttons to press.
    """
    keys = pygame.key.get_pressed()
    turn_left = False
    turn_right = False
    move_forward = False
    move_backward = False

    if keys[pygame.K_a]:
        turn_left = True
    if keys[pygame.K_d]:
        turn_right = True
    if keys[pygame.K_w]:
        move_forward = True
    if keys[pygame.K_s]:
        move_backward = True
    if keys[pygame.K_r]:
        player_car.reset()
        game_info.reset()

    # Player Car Actions
    if (not(move_forward) and not(move_backward)) or (move_forward and move_backward):
        if turn_left and not(turn_right):
            # 1 - just left
            action_no = 1
        elif not(turn_left) and turn_right:
            # 2 - just right
            action_no = 2
        else:
            # 9 - do nothing
            action_no = 9
    elif move_forward and not(move_backward):
        if turn_left and not(turn_right):
            # 5 - forward left
            action_no = 5
        elif not(turn_left) and turn_right:
            # 6 - forward right
            action_no = 6
        else:
            # 3 - just forward
            action_no = 3
    elif not(move_forward) and move_backward:
        if turn_left and not(turn_right):
            # 7 - backward left
            action_no = 7
        elif not(turn_left) and turn_right:
            # 8 - backward right
            action_no = 8
        else:
            # 4 - just backward
            action_no = 4

    return action_no


def handle_collision(player_car):
    bounce_flag = 0
    if MANUAL_CONTROL:
        if player_car.collide(TRACK_BORDER_MASK) != None:
            player_car.bounce()
            # Add a delay after bounce (where no input allowed)
            bounce_flag = 6
    else:
        if player_car.collide(TRACK_BORDER_MASK) != None:
            player_car.dead = True
    return bounce_flag

run = True
clock = pygame.time.Clock()
images = [(BACKGROUND, (0,0)), (TRACK, (0,0))]
bounce_flag = 0

while run:
    clock.tick(FPS)
    draw(WIN, images, player_car, game_info)

    if MANUAL_CONTROL:
        action_no = detect_input()
    else:
        # Get input from AI
        action_no = 9 # Do nothing currently

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    if bounce_flag > 0:
        # Overwrite player input to "Do Nothing"
        bounce_flag -= 1
        action_no = 9

    player_car.take_action(action_no)

    distances = beam_sensors.beam_distances(player_car)

    if bounce_flag == 0:
        bounce_flag = handle_collision(player_car)

    pygame.display.update()

    model_input = distances.append(player_car.vel)

pygame.quit()
