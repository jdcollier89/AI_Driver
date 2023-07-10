import pygame
import math

from src.utils import blit_rotate_center, scale_image

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG # Pickup img from child
        self.max_vel = max_vel
        self.rotation_vel = rotation_vel
        self.acceleration = 0.1
        self.drift_fric = 0.8 # Amount to reduce drift by each tick
        self.reset()

    def reset(self):
        self.angle = 0
        self.x, self.y = self.START_POS
        self.rot_x, self.rot_y = (0, 0)
        self.rot_img = self.img
        self.vel = 0

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        self.rot_img, (self.rot_x, self.rot_y) = blit_rotate_center(
            win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backwards(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)

        vel_x = self.vel * math.sin(radians)
        vel_y = self.vel * math.cos(radians)

        self.x -= vel_x
        self.y -= vel_y

    def reduce_speed(self):
        if self.vel < 0:
            self.vel = min(self.vel + self.acceleration/2, 0)
        else:
            self.vel = max(self.vel - self.acceleration/2, 0)

        self.move()

    def collide(self, mask, x=0, y=0):
        # x, y parameters are pos of input mask
        car_mask = pygame.mask.from_surface(self.rot_img)
        offset = (int(self.rot_x-x), int(self.rot_y-y))
        poi = mask.overlap(car_mask, offset)
        return poi
    
    def bounce(self):
        self.vel = -self.vel/2
        self.move()

        
class PlayerCar(AbstractCar):
    IMG = scale_image(pygame.image.load("imgs/grey-car.png"), 0.55)
    START_POS = (170, 200)