import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2 as cv

#TensorFlow - Importing the Libraries
import tensorflow as tf
from tensorflow import keras

# Loading the Data
# training directory - should have sub directories named after each label
DATADIR = "C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\TRAIN" 
TESTDIR = "C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\TEST" # test directory
LABELS = ["Class 1", "Class 2"] # should be filled with our determined labels

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
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (IMG_LENGTH, IMG_WIDTH))
            train_images_tf.append(new_array) # this is not running
            train_labels_tf = np.append(train_labels_tf, class_num)
        except Exception as e:
            pass

train_images_tf = np.array(train_images_tf).reshape(-1, IMG_LENGTH, IMG_WIDTH,3)

# Building the Model - replace each Conv2D parameter as necessary to fit
#   amount of images we use to train, and other factors
# Change the input_shape parameter to have the dimensions of the image not 128
modeltf = keras.Sequential([
    keras.layers.Conv2D(input_shape=(256,256,3), filters=32, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation='sigmoid') # 2 should be replaced with number of labels, sigmoid while binary output
])


# Visualizing the Model
modeltf.compile(loss='sparse_categorical_crossentropy', #keras.losses.binary_crossentropy, # binary while output layer is 2
              optimizer='adam',
              metrics=['accuracy'])
modeltf.summary()


# Training the Model
#train_images_tensorflow = (train_images_tf / 255.0).reshape(train_images_tf.shape[0], 28, 28, 1)
#test_images_tensorflow = (test_images_tf / 255.0).reshape(test_images_tf.shape[0], 28, 28 ,1)
#train_labels_tensorflow=keras.utils.to_categorical(train_labels_tf)
#test_labels_tensorflow=keras.utils.to_categorical(test_labels_tf)

print("Training the model")
modeltf.fit(train_images_tf, train_labels_tf, epochs=7, batch_size=5)


os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Comparing the Results
for img in os.listdir(TESTDIR):
    print("starting image process")
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        new_img = cv.resize(img_array, (IMG_LENGTH, IMG_WIDTH))
        new_shape = new_img.reshape(-1, IMG_LENGTH, IMG_WIDTH, 3)
        predictions = modeltf.predict(new_shape)
        plt.imshow(new_img)
        plt.show()
        print(predictions)
        print(LABELS[np.argmax(predictions)])
    except Exception as e:
        pass

# Save the model for future use
#modeltf.save('simpleCNN_MNIST')
