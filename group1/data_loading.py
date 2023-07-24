from skimage import io
import numpy as np
import csv
import os
from interactive_kit import imviewer as viewer

import time
import cv2

def list_images(dir_path):
    """
    List all files in a directory.
    dir_path: path to the directory
    returns: a list of the files in the directory
    """
    images = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".tif"):
            images.append(filename)
    return images


def load_image(path, max_frames=999):
    """
    Load a .tif file containing multiple frames.
    path: path to the .tif file
    max_frames: Number of frames to read. 999 for all frames.
    returns: images : [frame, width, height, channel]
    """
    t1 = time.time()
    
    images = []
    if max_frames < 999:
        ret, images = cv2.imreadmulti(mats=images,
                                    filename=path,
                                    start=0,
                                    count=max_frames,
                                    flags=cv2.IMREAD_UNCHANGED)
    else:
        ret, images = cv2.imreadmulti(mats=images,
                                    filename=path,
                                    flags=cv2.IMREAD_UNCHANGED)
        

    # [nChannels x frames, width, height] 
    # -> [frames, channel, width, height] 
    # -> [frames, width, height, channel]
    results2 = np.moveaxis(np.array(images).reshape(-1,2,images[0].shape[0],images[0].shape[1]), 1, -1)

    # Normalize frames
    min = results2.min(axis=(1, 2), keepdims=True)
    max = results2.max(axis=(1, 2), keepdims=True)
    results2 = (results2-min) / (max-min)
    
    tmp = np.zeros((results2.shape[0], results2.shape[1], results2.shape[2], 3))
    tmp[:, :, :, :2] = results2
    results2 = tmp

    return results2


def load_csv(path):
    """
    Load a csv file containing metadata.
    path: path to the csv file
    returns: a list of the rows in the file
    """
    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


def load_data(dir_path):
    """
    Load all data from a directory containing .tif and .csv files.
    dir_path: path to the directory containing the data
    channels: number of channels in the .tif files
    returns: a list of images, a list of images names and metadata
    """
    images = []
    names = []
    metadata = []
    for filename in os.listdir(dir_path):

        if filename.endswith(".tif"):
            images.append(load_image(dir_path + "/" + filename))
            names.append(filename)

        elif filename.endswith(".csv"):
            metadata.append(load_csv(dir_path + "/" + filename))

    return images, names, metadata