import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random

#TensorFlow - Importing the Libraries
import tensorflow as tf
from tensorflow import keras

# Loading the Data
# training directory - should have sub directories named after each label
DATADIR = "C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\TRAIN" 
TESTDIR = "C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\TEST" # test directory
LABELS = ["Class 1", "Class 2", "Class 3", "Class 6", "Class 7", "Class 8"] # should be filled with our determined labels

train_images_tf = []
train_labels_tf = np.empty(0)

# Shaping Input - change to match dimensions of processed images
IMG_LENGTH = 256
IMG_WIDTH = 256

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
            
            train_images_tf.append(new_array) 
            train_labels_tf = np.append(train_labels_tf, class_num)
        except Exception as e:
            pass

train_images_tf = np.array(train_images_tf).reshape(-1, IMG_LENGTH, IMG_WIDTH, 1)

# Building the Model - replace each Conv2D parameter as necessary to fit
#   amount of images we use to train, and other factors
# Change the input_shape parameter to have the dimensions of the image not 128
modeltf = keras.Sequential([
    keras.layers.Conv2D(input_shape=(IMG_LENGTH,IMG_WIDTH,1), filters=32, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(64, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(6, activation='softmax') # 2 should be replaced with number of labels, sigmoid while binary output
])


# Visualizing the Model
adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
modeltf.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
modeltf.summary()


# Training the Model

print("Training the model")
modeltf.fit(train_images_tf, train_labels_tf, epochs=90, batch_size=64)


os.environ['KMP_DUPLICATE_LIB_OK']='True' # temporary fix for duplicate file issue
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.

# Comparing the Results
for img in os.listdir(TESTDIR):
    print("starting image process")
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        if len(img_array.shape) > 2:
            img_array = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        if random.random() > 0:
            kernel = np.array([[0,-1,0], 
                               [-1,5,-1], 
                               [0,-1,0]])
            img_array = cv.filter2D(img_array, -1, kernel)
        img_array = cv.resize(img_array, (IMG_LENGTH, IMG_WIDTH))
        new_array = cv.normalize(img_array, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
        new_shape = new_array.reshape(-1, IMG_LENGTH, IMG_WIDTH, 1)
        predictions = modeltf.predict(new_shape)
        plt.imshow(new_array)
        plt.show()
        print("image: " + img)
        print(predictions)
        print(LABELS[np.argmax(predictions)])
    except Exception as e:
        pass

# Save the model for future use
#modeltf.save('simpleCNN_MNIST')
#modeltf.save('CNN_Version1_3Classes')
#modeltf.save('CNN_Version2_3Classes')
#modeltf.save('CNN_Version3_4Classes')