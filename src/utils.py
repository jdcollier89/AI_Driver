import pygame
import math


def scale_image(img, factor):
    """
    Scale a pygame image by a multiplier (e.g. 0.9 for image 90% size)
    """
    size = round(img.get_width()*factor), round(img.get_height()*factor)
    return pygame.transform.scale(img, size)

def blit_rotate_center(win, image, center, angle, display=True):
    """
    Rotate an image around its center point.
    Returns the rotated image and its top left coords.
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(center=center).center)
    win.blit(rotated_image, new_rect.topleft)
    
    return rotated_image, new_rect.topleft

def distance_between_points(point1, point2):
    """
    Given the coordinates of two points (given as a tuple),
    return the distance between the two points.
    """
    try:
        x_dist = abs(point1[0]-point2[0])
        y_dist = abs(point1[1]-point2[1])
        return round(math.hypot(x_dist, y_dist), 1)
    except:
        return None