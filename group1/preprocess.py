import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.filters as skimage_filters
from skimage.segmentation import inverse_gaussian_gradient

def standardize(image_sequence):
    """Standardize image values

    Args:
        image_sequence: [frame, width, height, channels]
    """
    for frame in range(image_sequence.shape[0]):
        for color in range (image_sequence.shape[3]):
            std = np.std(image_sequence[frame,:,:,color])
            avg = np.mean(image_sequence[frame,:,:,color])
            if std>0 :
                # Adjust pixel values so that they have a std deviation of 0.25 and an average of 0
                image_sequence[frame,:,:,color] = (image_sequence[frame,:,:,color] - avg) / (4*std)
                # Apply sigmoid to map values from [-1,1]to [0,1]
                image_sequence[frame,:,:,color] = 1 / (1 + np.exp(-4*image_sequence[frame,:,:,color]))


def edge_detection(image, plot_results=False):
    """Filter and do edge detection on image

    Args:
        image (np.ndarray): Image of cells [frame, width, height] (Only one channel!)
        plot_results (bool, optional): Plot intermediate results for first frame

    Returns:
        np.ndarray [frame, width, height]: Cell and bleb edges in input image
    """
    # Only one frame
    one_frame = False
    if len(image.shape) == 2:
        one_frame = True
        image = image[np.newaxis, ...]

    for frame in range(image.shape[0]):

        # Blur and equalize
        img = image[frame,:,:]
        # img_sharp_i = sharpen(1-img)
        # img_log = LOGfilter(img, 5, 1)
        # img_lapl = np.abs(cv2.Laplacian((img_sharp*255).astype(np.uint8), cv2.CV_64F))
        # img_median = cv2.medianBlur(np.float32(img), ksize=3)
        # img_gaussian = cv2.GaussianBlur(img_median,(9,9),cv2.BORDER_DEFAULT)
        # print(f'{img_gaussian.max() = }')
        # img_equalhist = cv2.equalizeHist((img_gaussian*255).astype(np.uint8))
        img_sharp = sharpen(img)
        # img_equalhist = cv2.equalizeHist((img*255).astype(np.uint8)).astype(np.float32)/255.

        # img_eq_inv = inverse_gaussian_gradient(img_equalhist)
        # img_inv = inverse_gaussian_gradient(img)
        # img_inv_sharp = inverse_gaussian_gradient(img_sharp)

        # blur1 = skimage_filters.gaussian(img, 2)
        # sob1 = skimage_filters.sobel(blur1)
        # blur2 = skimage_filters.gaussian(1-img, 2)
        # sob2 = skimage_filters.sobel(blur2)
        # Canny edge detection
        # image_canny = cv2.Canny((img_sharp*255).astype(np.uint8), 100, 256)

        # Blur edges - thought if might help the active contour algo
        # image[frame,:,:] = image_canny
        # image[frame,:,:] = cv2.GaussianBlur(image_canny,(31,31),cv2.BORDER_DEFAULT)
        # image[frame,:,:] = sharpen(sharpen(img_inv_sharp))
        image[frame,:,:] = img_sharp

        # Plot results
        if plot_results and frame==0:
            # fig, axs = plt.subplots(2,2,figsize=(10,20))
            # axs = axs.ravel()
            # for ax in axs: ax.set_xticks([]); ax.set_yticks([])
            # axs[0].imshow(sharpen(img), cmap='gray')
            # axs[1].imshow(img_sharp, cmap='gray')
            # axs[2].imshow(img_inv_sharp, cmap='gray')
            # axs[3].imshow(img_inv, cmap='gray')
            # axs[1].imshow(img_equalhist, cmap='gray')
            # axs[2].imshow(blur, cmap='gray')
            # axs[3].imshow(img, cmap='gray')

            plt.tight_layout()
            plt.show()

    if one_frame:
        return image[0,:,:]
    else:
        return image
    

# Try to find edges in image

from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

def sharpen(img, threshold = 0.25, cut=0.1):
    """Sharpen image and remove directional shadows

    Args:
        img: [width, height, channels]
        threshold: Cut pixels outside. Defaults to 0.25.
        cut: Remove background. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    
    img_blur = (3*img + gaussian_filter(img, 3))/4 #lowpass to diminish noise
    # cut the pixels below threshold and above 1-threshold for computing the mean 
    mean = np.mean(img_blur[(img_blur >= threshold) & (1-img_blur >= threshold)])
    
    lower = 1 - np.where(img_blur < mean, img_blur, 1) # shadows correspond to SW edges, invert them
    upper = np.where(img_blur >= mean, img_blur, mean) # lights correspond to NE edges, take them as is

    lower = (lower - mean) /(1 - mean) # normalize
    upper = (upper - mean) /(1 - mean) # normalize

    lower = np.where(lower>cut,lower,0) # remove background
    upper = np.where(upper>cut,upper,0) # remove background
    
    i = lower + upper # put edges together again
    
    # i = cv2.medianBlur(np.float32(img_abs), ksize=1)
    i= 2*i -1

    i = 1 / (1 + np.exp(-4*i)) # sigmoid, to map values to [0:1] smoothly
    
    return i


def LOGfilter(src_img, k_size, sigma):
    gaussian_img = cv2.GaussianBlur(src_img, (k_size, k_size), sigma)
    log_img = cv2.Laplacian(gaussian_img, ddepth=cv2.CV_32F, ksize=k_size)
    log_img[log_img > 0] = 0
    log_img = log_img * -1
    log_img = abs(log_img)
    log_min = np.min(log_img)
    log_max = np.max(log_img)
    log_nlz = (log_img - log_min) / (log_max - log_min) * 255
    log_nlz = log_nlz.astype(np.uint8)
    return log_nlz