import pygame
import math

from src.utils import rotate_center, scale_image

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG.convert() # Pickup img from child
        self.max_vel = max_vel
        self.rotation_vel = rotation_vel
        self.acceleration = 0.2
        self.driftFriction = 0.75 # Amount to reduce drift by each tick
        self.driftMomentum = 0
        self.reset()

    def reset(self):
        self.angle = 0
        self.x, self.y = self.START_POS # Define center of car
        self.update_car_img() # Update rotated image based on new position
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
        
    def update_car_img(self):
        """
        Calculate the new image/position of car based on current rotation
        """
        self.rot_img, (self.rot_x, self.rot_y) = rotate_center(
                    self.img, (self.x, self.y), self.angle)

    def draw(self, win):
        """
        Draw the car on screen
        """
        win.blit(self.rot_img, (self.rot_x, self.rot_y))

    def move_forward(self, turn_left, turn_right):
        """
        Apply forward acceleration and move car
        """
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move(turn_left, turn_right)

    def move_backwards(self, turn_left, turn_right):
        """
        Apply reverse acceleration and move car
        """
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
        """
        Apply friction if no acceleration is provided.
        Half speed every tick.
        """
        if self.vel < 0:
            self.vel = min(self.vel + self.acceleration/2, 0)
        else:
            self.vel = max(self.vel - self.acceleration/2, 0)

        self.move(turn_left, turn_right)

    def collide(self, mask, x=0, y=0):
        """
        Detect if collision has occured between provided mask and the car object.
        """
        # x, y parameters are pos of input mask
        car_mask = pygame.mask.from_surface(self.rot_img)
        offset = (int(self.rot_x-x), int(self.rot_y-y))
        poi = mask.overlap(car_mask, offset)
        car_mask = None # Deallocate for memory
        return poi
    
    def bounce(self):
        """
        Reverse the direction of car if person controlling crashes into barrier.
        """
        self.vel = -self.vel/2
        self.driftMomentum = -self.driftMomentum

        
class PlayerCar(AbstractCar):
    IMG = scale_image(pygame.image.load("imgs/grey-car.png"), 0.55)
    START_POS = (177, 245) #(175, 200)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.dead = False
        self.bounce_flag = 0

    def take_action(self, action_no):
        """
        Have the car move according to the associated action_no provided.
        Reduce speed (due to friction) of car if no forward or reverse acceleration.
        """
        if action_no == 1:
            # 1 - just left
            if self.vel > 0:
                self.rotate(left=True)
            self.reduce_speed(True, False)
        elif action_no == 2:
            # 2 - just right
            if self.vel > 0:
                self.rotate(right=True)
            self.reduce_speed(False, True)
        elif action_no == 3:
            # 3 - just forward
            self.move_forward(False, False)
        elif action_no == 4:
            # 4 - just backward
            self.move_backwards(False, False)
        elif action_no == 5:
            # 5 - forward left
            self.rotate(left=True)
            self.move_forward(True, False)
        elif action_no == 6:
            # 6 - forward right
            self.rotate(right=True)
            self.move_forward(False, True)
        elif action_no == 7:
            # 7 - backward left
            self.rotate(left=True)
            self.move_backwards(True, False)
        elif action_no == 8:
            # 8 - backward right
            self.rotate(right=True)
            self.move_backwards(False, True)
        elif action_no == 9:
            # 9 - do nothing
            self.reduce_speed(False, False)
