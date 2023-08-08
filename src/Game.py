import pygame

from src.utils import scale_image
from src.Cars import PlayerCar
from src.GameInfo import GameInfo
from src.Sensor import Sensor
from src.RewardGates import RewardGate

pygame.font.init()

MAIN_FONT = pygame.font.SysFont("comicsans", 40)

BACKGROUND = scale_image(pygame.image.load("imgs/green-grass-background.jpg"), 0.35)

TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)

TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Car Driving")

class Game:
    def __init__(self):
        self.MANUAL_CONTROL = False
        self.player_car = PlayerCar(6, 5)
        self.game_info = GameInfo()
        self.beam_sensors = Sensor(WIN, TRACK_BORDER)
        self.reward_gates = RewardGate()
        self.reward = 0

        self.clock = pygame.time.Clock()
        self.images = [(BACKGROUND, (0,0)), (TRACK, (0,0))]

    def draw(self):
        """
        Draw the background, text and player car to the screen
        """

        for img, pos in self.images:
            WIN.blit(img, pos)

        score_text = MAIN_FONT.render(f"Score: {self.game_info.score}", 1, (255, 255, 255))
        WIN.blit(score_text, (10, HEIGHT - score_text.get_height() - 70))

        time_text = MAIN_FONT.render(f"Time: {self.game_info.get_level_time()}s", 1, (255, 255, 255))
        WIN.blit(time_text, (10, HEIGHT - time_text.get_height() - 40))

        vel_text = MAIN_FONT.render(f"Vel: {round(self.player_car.vel, 1)}px/s", 1, (255, 255, 255))
        WIN.blit(vel_text, (10, HEIGHT - vel_text.get_height() - 10))

        WIN.blit(self.reward_gates.return_active(), (0,0))

        self.player_car.draw(WIN)

    def handle_collision(self):
        """
        Check for collision between car and track border
        """
        bounce_flag = 0

        if self.MANUAL_CONTROL:
            if self.player_car.collide(TRACK_BORDER_MASK) != None:
                self.player_car.bounce()
                # Add a delay after bounce (where no input allowed)
                bounce_flag = 6
        else:
            if self.player_car.collide(TRACK_BORDER_MASK) != None:
                self.player_car.dead = True

        return bounce_flag
    
    def game_reset(self):
        """
        Reset all elements of the game
        """
        self.player_car.reset()
        self.game_info.reset()
        self.reward_gates.reset()
    
    def detect_input(self):
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
            self.game_reset()

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

    def manual_loop(self):
        """
        Iterate the game state by one tick and detect player inputs.
        """

        run = True
        self.MANUAL_CONTROL = True

        action_no = self.detect_input()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        self.game_loop(action_no)

        return run
    
    def game_loop(self, action_no):
        """
        Given a set action number, iterate the game state by one tick.
        """
        self.reward = -1

        if self.player_car.bounce_flag > 0:
            # Overwrite player input to "Do Nothing"
            self.player_car.bounce_flag -= 1
            action_no = 9

        self.player_car.take_action(action_no)

        passed = self.reward_gates.passed_gate(self.player_car, self.game_info)
        if passed:
            self.reward = 10


        if self.player_car.bounce_flag == 0:
            self.player_car.bounce_flag = self.handle_collision()

        return
    
    def game_state(self):
        distances = self.beam_sensors.beam_distances(self.player_car)

        gate_dist = self.reward_gates.distance_to_gate(self.player_car.x, self.player_car.y)

        model_input = distances
        model_input.append(self.player_car.vel)
        model_input.append(gate_dist)

        done = self.game_finished()
        if done:
            self.reward = -100

        return model_input, self.reward, done
    
    def game_finished(self):
        return self.player_car.dead