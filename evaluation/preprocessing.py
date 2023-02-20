import numpy as np 
import os
import cv2 as cv
import random
from constants import *

#train_images_tf = []
#train_labels_tf = np.empty(0)

# Where directory DATADIR contains directories for each category in LABELS
# i.e. our training set is already labelled
for category in LABELS:
    path = os.path.join(DATADIR, category)
    #print(path)
    class_num = LABELS.index(category)
    #print(class_num)
    for img in os.listdir(path):
        try:
            # Read in Image
            img_array = cv.imread(os.path.join(path, img))
            
            # Convert to GrayScale
            if len(img_array.shape) > 2:
                img_array = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # Randomly apply a sharpen to the image
            if random.random() > 0:
                kernel = np.array([[0,-1,0], 
                                   [-1,5,-1],
                                   [0,-1,0]])
                img_array = cv.filter2D(img_array, -1, kernel)
            
            # Resize Image
            img_array = cv.resize(img_array, (IMG_LENGTH, IMG_WIDTH))
            
            # Normalize the Image
            new_array = cv.normalize(img_array, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
            
            # Save Image to Target Directory
            targetpath = os.path.join(PREPROCESSEDPATH, category)
            cv.imwrite(os.path.join(targetpath, img), new_array)
            
            #train_images_tf.append(new_array) 
            #train_labels_tf = np.append(train_labels_tf, class_num)
        except Exception as e:
            pass

#train_images_tf = np.array(train_images_tf).reshape(-1, IMG_LENGTH, IMG_WIDTH, 1)