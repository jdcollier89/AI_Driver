import pygame
import math

from src.utils import scale_image
from src.Cars import PlayerCar
from src.GameInfo import GameInfo
from src.utils import distance_between_points

START_GATE = 0

class RewardGate:
    def __init__(self):
        self.no_of_gates = 13
        # Provide list of reward gate image file
        self.reward_gate = [None] * self.no_of_gates
        for i in range(1, self.no_of_gates+1):
            fname = f"imgs/reward-gates/RewardGate{i}.png"
            self.reward_gate[i-1] = scale_image(pygame.image.load(fname), 0.9)
        # Set activate gate to first one
        self.reset()

    def return_active(self):
        # return img of activate gate
        return self.reward_gate[self.active_gate]
    
    def return_active_mask(self):
        # return mask of activate gate
        return pygame.mask.from_surface(self.return_active())
    
    def reset(self):
        """
        Reset the active gate to be the first gate again (if resetting game)
        """
        self.active_gate = START_GATE

    def increment_gate(self):
        self.active_gate += 1
        # Reset to first gate when full lap done
        if self.active_gate >= self.no_of_gates:
            self.active_gate = 0

    # return angle and distance to next gate
    def return_gate_posn(self):
        """
        Identify the central point of the current active reward gate.
        """
        return self.return_active_mask().centroid()
    
    def distance_to_gate(self, x, y):
        """
        Return the distance between given x,y coordinates and the next
        reward gate (e.g. current active one).
        """
        distance = distance_between_points(self.return_gate_posn(), (x,y))
        return distance
    
    def angle_to_gate(self, x, y):
        (gate_x, gate_y) = self.return_gate_posn()
        adj = gate_x - x
        opp = gate_y - y
        angle = math.degrees(math.atan(adj/opp))

        return angle

    def passed_gate(self, player_car, game_info):
        """
        Check whether the player car has passed the currently
        active reward gate. Increment score if so.
        """
        gate_mask = self.return_active_mask()
        if player_car.collide(gate_mask) != None:
            game_info.score += 1
            self.increment_gate()
            return True
        else:
            return False