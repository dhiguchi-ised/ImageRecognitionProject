import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2 as cv

# Shaping Input - change to match dimensions of processed images
IMG_LENGTH = 256
IMG_WIDTH = 256

# Read in Image
img_array = cv.imread('C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\TRAIN\\Class 8\\CA-2704375-REPRESENTATIVE_DRAWING-20220201-6.tif')

# Convert to GrayScale
if len(img_array.shape) > 2:
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

# Randomly apply a sharpen to the image
kernel = np.array([[0,-1,0], 
                   [-1,4.75,-1],
                   [0,-1,0]])
img_array = cv.filter2D(img_array, -1, kernel)

# Normalize the Image
img_array = cv.normalize(img_array, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F) 

# Resize Image
img_array = cv.resize(img_array, (IMG_LENGTH, IMG_WIDTH))

plt.imshow(img_array)
plt.show()
cv.imwrite('C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\general scripts\\SharpenCheck\\resizeafter.tif', img_array)