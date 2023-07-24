import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_area(coordinates):
    """Calculate area inside contour

    Args:
        coordinates: List of coordinates [[y,x],[y,x],...]

    Returns:
        int: Area
    """
    return cv2.contourArea(coordinates.astype(np.float32))

def calculate_areas(snakes):
    """Calculate areas over time for all snakes

    Args:
        snakes: Dictionary of all snakes
    """

    for snake in snakes:
        snake['area'] = []

        for frame in range(snake.get('data').shape[0]):
            a_cv = calculate_area(snake.get('data')[frame])

            snake['area'].append(a_cv)