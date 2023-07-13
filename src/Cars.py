import pygame
import math

from src.utils import blit_rotate_center, scale_image

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG # Pickup img from child
        self.max_vel = max_vel
        self.rotation_vel = rotation_vel
        self.acceleration = 0.1
        self.driftFriction = 0.75 # Amount to reduce drift by each tick
        self.driftMomentum = 0
        self.reset()

    def reset(self):
        self.angle = 0
        self.x, self.y = self.START_POS # Define center of car
        self.rot_x, self.rot_y = (0, 0) # Top left of rotated image
        self.rot_img = self.img
        self.vel = 0
        self.driftMomentum = 0

    def rotate(self, left=False, right=False):
        if abs(self.vel) > 5:
            multiplier = abs(self.vel)/5
        else:
            multiplier = 1

        if left:
            self.angle += self.rotation_vel * multiplier
        elif right:
            self.angle -= self.rotation_vel * multiplier

    def draw(self, win):
        self.rot_img, (self.rot_x, self.rot_y) = blit_rotate_center(
            win, self.img, (self.x, self.y), self.angle)

    def move_forward(self, turn_left, turn_right):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move(turn_left, turn_right)

    def move_backwards(self, turn_left, turn_right):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move(turn_left, turn_right)

    def move(self, turn_left, turn_right):
        radians = math.radians(self.angle)
        drift_radians = math.radians(self.angle + 90)

        driftAmount = self.vel * self.rotation_vel / (9.0 * 8.0)
        if self.vel < 5:
            driftAmount = 0

        if turn_left:
            self.driftMomentum -= driftAmount
        if turn_right:
            self.driftMomentum += driftAmount

        vel_x = self.vel * math.sin(radians)
        vel_y = self.vel * math.cos(radians)

        drift_x = self.driftMomentum * math.sin(drift_radians)
        drift_y = self.driftMomentum * math.cos(drift_radians)

        self.x -= (vel_x + drift_x)
        self.y -= (vel_y + drift_y)

        self.driftMomentum *= self.driftFriction

    def reduce_speed(self, turn_left, turn_right):
        if self.vel < 0:
            self.vel = min(self.vel + self.acceleration/2, 0)
        else:
            self.vel = max(self.vel - self.acceleration/2, 0)

        self.move(turn_left, turn_right)

    def collide(self, mask, x=0, y=0):
        # x, y parameters are pos of input mask
        car_mask = pygame.mask.from_surface(self.rot_img)
        offset = (int(self.rot_x-x), int(self.rot_y-y))
        poi = mask.overlap(car_mask, offset)
        return poi
    
    def bounce(self):
        self.vel = -self.vel/2
        self.driftMomentum = -self.driftMomentum
        self.move(False, False)

        
class PlayerCar(AbstractCar):
    IMG = scale_image(pygame.image.load("imgs/grey-car.png"), 0.55)
    START_POS = (175, 200)