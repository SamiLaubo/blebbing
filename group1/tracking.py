import numpy as np
import cv2

from skimage.segmentation import active_contour
from area import calculate_area
from tqdm import tqdm


def track(image, snakes):
    """
    Iterates on all frames of the image to deduce the border
    At each new frame, we take the borders of the previous frame
    and adapt them to the new frame using track_next
    :param image: the image [frame, width, height]
    :param initial_borders: initial boolean image of each bleb [width, height, bleb]
    :return: the blebs at each frame [frame, width, height, bleb] : boolean image of the blebs
    """

    for idx, snake in enumerate(snakes): # iterate on each snake
        print(f'Bleb {idx+1}/{len(snakes)}')

        init_snake = smooth_snake(snake.get('data'))

        snake_frames = image.shape[0] - snake.get('start_frame')
        snake['data'] = np.zeros((snake_frames, init_snake.shape[0], init_snake.shape[1]))
        snake['data'][0] = track_next(image[snake.get('start_frame')], init_snake)

        snake_idx = 1
        for frame in tqdm(range(snake.get('start_frame')+1, image.shape[0])):  # iterate on each frame
            
            snake['data'][snake_idx] = track_next(image[frame], snake.get('data')[snake_idx - 1], image[frame-1])

            snake_idx += 1

def track_next(frame, snake, prev_frame=None):
    """From the border of a bleb in the current frame, compute the next frame's border

    Parameters:
        frame: [width, height]
        snake: List of coordinates [[y,x],[y,x],...]

    Returns:
        List of coordinates [[y,x],[y,x],...] for new snake
    """
    snake_temp = snake.copy()

    # Crop image to only look at area around bleb
    y_min, x_min = snake.min(axis=0).astype(np.int32) - 30
    y_max, x_max = snake.max(axis=0).astype(np.int32) + 30
    min_crop = np.array([y_min, x_min])

    piece = frame[y_min:y_max, x_min:x_max]

    if prev_frame is not None: # Not first frame
        max_num_iter = 100
        first_frame_mult = 1
    else: # First frame
        max_num_iter = 100
        first_frame_mult = .1

    # Parameters for active contour
    alpha = 0.01
    beta = 10.
    w_edge = 1.5
    w_line = 1.

    # Fit snake to bleb
    snake_next = active_contour(
        piece, smooth_snake(snake_temp) - min_crop,
        max_num_iter=max_num_iter, 
        alpha=alpha, beta=beta,
        w_edge=w_edge,
        w_line=w_line
    ) + min_crop

    # Too sharp drop in area or circularity: try with larger dilation and translation
    i = 1
    while (calculate_area(snake_next) < calculate_area(snake) - 100 * first_frame_mult) or circularity(snake) < .7:
        snake_next = active_contour(
            piece, dilate_snake(smooth_snake(snake_temp, 0.1), (10+i)/10) - min_crop, 
            max_num_iter=max_num_iter,
            alpha=alpha, beta=beta,
            w_edge=w_edge,
            w_line=w_line
        ) + min_crop

        i += 1
        if i > 10:
            break

    # Return smoothed fitted snake
    return smooth_snake(snake_next, 0.03)

def dilate_snake(snake, scale=1.1, frame_prev=None, frame=None):
    """ Expand all points in contour radially

    Parameters:
        snake: List of coordinates [[y,x],[y,x],...]
        scale: scaling factor

    Returns:
        scales snake [[y,x],[y,x],...]
    """

    if frame is not None and frame_prev is not None:
        y_min, x_min = snake.min(axis=0).astype(np.int32)
        y_max, x_max = snake.max(axis=0).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min

        y_min -= h//2 
        x_min -= w//2 
        y_max += h//2
        x_max += w//2 

        piece_prev = frame_prev[y_min:y_max, x_min:x_max]
        piece = frame[y_min:y_max, x_min:x_max]
    
        # Don't expand if bleb does not expand
        if piece_prev.sum() >= piece.sum():
            return snake

    cx, cy = np.mean(snake, axis=0)

    snake_scaled = snake - [cx, cy]
    snake_scaled *= scale
    snake_scaled += [cx, cy]

    return snake_scaled


def smooth_snake(snake, cutoff=.01):
    """Smooth and create circular points from snake

    Args:
        snake: List of coordinates [[y,x],[y,x],...]
        cutoff: Cutoff frequency

    Returns:
        List of coordinates [[y,x],[y,x],...]
    """
    # Create Fourier signal
    signal = snake[:,1].T + 1j*snake[:,0].T
    
    # FFT and frequencies
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])

    # Filter
    fft[np.abs(freq) > cutoff] = 0

    # IFFT
    signal_ifft = np.fft.ifft(fft)

    snake_smooth = np.zeros(snake.shape)
    snake_smooth[:,1] = np.real(signal_ifft)
    snake_smooth[:,0] = np.imag(signal_ifft)

    return snake_smooth

import matplotlib.pyplot as plt
import scipy.ndimage as ndi
def translate_with_flow(frame_prev, frame, snake):
    """Translate snake with general flow in cropped image

    Args:
        frame_prev: _description_
        frame: _description_
        snake: List of coordinates [[y,x],[y,x],...]
        mult (_type_, optional): _description_. Defaults to 1..

    Returns:
        _type_: _description_
    """
    y_min, x_min = snake.min(axis=0).astype(np.int32)
    y_max, x_max = snake.max(axis=0).astype(np.int32)

    h, w = y_max - y_min, x_max - x_min

    y_min -= h//2 
    x_min -= w//2 
    y_max += h//2
    x_max += w//2 

    piece_prev = frame_prev[y_min:y_max, x_min:x_max]
    piece = frame[y_min:y_max, x_min:x_max]

    c_prev = np.array(ndi.center_of_mass(piece_prev))
    c = np.array(ndi.center_of_mass(piece))

    flow_vec = (c - c_prev)

    return snake + flow_vec

def circularity(snake):
    """
    Calculate circularity of shape
    """
    area = cv2.contourArea(snake.astype(np.float32))
    perimeter = cv2.arcLength(snake.astype(np.float32), closed=True)

    return 4*np.pi*area/np.power(perimeter, 2)