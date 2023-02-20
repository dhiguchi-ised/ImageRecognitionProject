import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random

#TensorFlow - Importing the Libraries
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\models\\CNN_Version3_4Classes')
TESTDIR = "C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\TEST"
IMG_LENGTH = 256
IMG_WIDTH = 256
LABELS = ["Class 1", "Class 2", "Class 3", "Class 8"]

for img in os.listdir(TESTDIR):
    print("starting image process")
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        if len(img_array.shape) > 2:
            img_array = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        if random.random() > 0:
            kernel = np.array([[0,-1,0], 
                               [-1,4.75,-1], 
                               [0,-1,0]])
            img_array = cv.filter2D(img_array, -1, kernel)
        new_array = cv.resize(img_array, (IMG_LENGTH, IMG_WIDTH))
        new_array = cv.normalize(new_array, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
        new_shape = new_array.reshape(-1, IMG_LENGTH, IMG_WIDTH, 1)
        predictions = model.predict(new_shape)
        #plt.imshow(new_img)
        #plt.show()
        print("image: " + img)
        print(predictions)
        print(LABELS[np.argmax(predictions)])
    except Exception as e:
        pass