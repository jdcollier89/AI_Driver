import pygame
import math
from src.utils import distance_between_points

DEBUG_SENSORS = True

class Sensor:
    def __init__(self, surface, track_border):
        self.surface = surface
        self.WIDTH = surface.get_width()
        self.HEIGHT = surface.get_height()

        # Surface to contain sensor beams
        self.beam_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

        track_border_mask = pygame.mask.from_surface(track_border)
        track_border_mask_fx = pygame.mask.from_surface(pygame.transform.flip(track_border, True, False))
        track_border_mask_fy = pygame.mask.from_surface(pygame.transform.flip(track_border, False, True))
        track_border_mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(track_border, True, True))
        self.flipped_masks = [[track_border_mask, track_border_mask_fy], [track_border_mask_fx, track_border_mask_fx_fy]]

    def draw_beam(self, angle, pos):
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))

        flip_x = c < 0
        flip_y = s < 0
        flipped_mask = self.flipped_masks[flip_x][flip_y]
        
        # compute beam final point
        x_dest = self.WIDTH * abs(c)
        y_dest = self.HEIGHT * abs(s)

        # clear beam surface in prep for new beam
        self.beam_surface.fill((0, 0, 0, 0))

        # draw a single beam to the beam surface based on computed final point
        pygame.draw.line(self.beam_surface, (0, 0, 255), (0, 0), (x_dest, y_dest))
        beam_mask = pygame.mask.from_surface(self.beam_surface)

        # find overlap between "global mask" and current beam mask
        offset_x = self.WIDTH-1 - pos[0] if flip_x else pos[0]
        offset_y = self.HEIGHT-1 - pos[1] if flip_y else pos[1]
        hit = flipped_mask.overlap(beam_mask, (round(offset_x), round(offset_y)))
        if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
            hx = self.WIDTH-1 - hit[0] if flip_x else hit[0]
            hy = self.HEIGHT-1 - hit[1] if flip_y else hit[1]
            hit_pos = (hx, hy)

            if DEBUG_SENSORS:
                pygame.draw.line(self.surface, (0, 0, 255), pos, hit_pos)
                pygame.draw.circle(self.surface, (0, 255, 0), hit_pos, 3)
            return hit_pos
        else:
            return None
    
    # def distance_between_points(self, collision_point, origin):
    #     try:
    #         x_dist = abs(collision_point[0]-origin[0])
    #         y_dist = abs(collision_point[1]-origin[1])
    #         return round(math.hypot(x_dist, y_dist), 1)
    #     except:
    #         return None

    def beam_distances(self, player_car):
        origin = (player_car.x, player_car.y)

        # Beams are named relative to car orientation (e.g. north is front facing)
        e_beam = self.draw_beam(-player_car.angle, origin)
        se_beam = self.draw_beam(-player_car.angle + 45, origin)
        s_beam = self.draw_beam(-player_car.angle + 90, origin)
        sw_beam = self.draw_beam(-player_car.angle + 135, origin)
        w_beam = self.draw_beam(-player_car.angle + 180, origin)
        nw_beam = self.draw_beam(-player_car.angle + 225, origin)
        n_beam = self.draw_beam(-player_car.angle + 270, origin)
        ne_beam = self.draw_beam(-player_car.angle + 315, origin)

        distance_array = [
            distance_between_points(n_beam, origin), distance_between_points(ne_beam, origin), 
            distance_between_points(e_beam, origin), distance_between_points(se_beam, origin),
            distance_between_points(s_beam, origin), distance_between_points(sw_beam, origin),
            distance_between_points(w_beam, origin), distance_between_points(nw_beam, origin)
            ]
        
        return distance_array

