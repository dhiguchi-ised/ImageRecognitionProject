from constants import *
import os
import numpy as np
import cv2 as cv

#TensorFlow - Importing the Libraries
import tensorflow as tf
from tensorflow import keras

train_images_tf = []
train_labels_tf = np.empty(0)

for category in LABELS:
    path = os.path.join(PREPROCESSEDPATH, category)
    class_num = LABELS.index(category)
    for img in os.listdir(path):
        img_array = cv.imread(os.path.join(path, img))
        train_images_tf.append(img_array) 
        train_labels_tf = np.append(train_labels_tf, class_num)

train_images_tf = np.array(train_images_tf).reshape(-1, IMG_LENGTH, IMG_WIDTH, 1)

# Building the Model - replace each Conv2D parameter as necessary to fit
#   amount of images we use to train, and other factors
# Change the input_shape parameter to have the dimensions of the image not 128
modeltf = keras.Sequential([
    keras.layers.Conv2D(input_shape=(IMG_LENGTH,IMG_WIDTH,1), filters=32, kernel_size=7, strides=1, padding="same", activation=tf.nn.relu),
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
adam = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
modeltf.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
modeltf.summary()


# Training the Model

print("Training the model")
modeltf.fit(train_images_tf, train_labels_tf, epochs=55, batch_size=32)

# Save the Model
#modeltf.save('CNN_Version4_6Classes')
