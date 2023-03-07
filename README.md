# Purpose of Project
To explore the uses of machine learning when applied to patent images. We are currently using a Convolutional Neural Network to perform image classification on images submitted by patent applicants to
represent their inventions. This image classification would be able to help patent examiners quickly get results on already existing visually similar patent images.

# Description of Data
The data we are training our model with are patent images that we preprocess so that they are all 256x256 pixels, gray-scale, sharpened, and normalized. The complete set of patent data we plan to work consists of roughly 50000 images, but as part of our trained set, there are currently only ~300 images we have labelled (currently 6 classifications). We plan to continue to expand our training set.

Our test/verification set currently consists of 20 images.

# Description of Current CNN Model 
Currently takes in images with dimensions 256 x 256, that are gray-scale. 

Input layer has 32 filters, with a 5x5 kernel that performs 2 pixel strides. The first hidden layer consists of 64 filters, with a 5x5 kernel that performs 1 pixel strides. The second hidden layer consists of 128 filters, with a 3x3 kernel that performs 2 pixel strides. The third hidden layer consists of 64 filters, with a 3x3 kernel that performs 1 pixel strides. 
Right before our output layer we have a fully connected layer with 128 filters. We output 6 different labels. Between each covolutional layer, there is a max-pooling layer that is 2x2 with strides of 2.

The model is currently trained in random batches of size 64, and the process is set to 30 epochs. We are using the Adam optimizer with a learning rate of 0.001.

# Description of CNN Model Before Run 1 of Training Model Log
Currently takes in images with dimensions 256 x 256, that are gray-scale. 

Input layer has 32 filters, with a 7x7 kernel that performs 1 pixel strides. The first hidden layer consists of 64 filters, with a 5x5 kernel that performs 1 pixel strides. The second hidden layer consists of 128 filters, with a 3x3 kernel that performs 1 pixel strides. The third hidden layer consists of 64 filters, with a 3x3 kernel that performs 1 pixel strides. 
Right before our output layer we have a fully connected layer with 128 filters. We output 6 different labels.

The model is currently trained in random batches of size 32, and the process is set to 55 epochs.

# Using and/or Modifying the Model
There are three parts that collectively make up the process of using this convolutional neural network. There is the preprocessing.py file meant to preprocess all the patent images before being passed into the
CNN model, then we have the training.py file where the model will learn all the features and patterns it needs, and lastly the prediction.py file where we load in the trained model and can use it to classify
different images that we give it based on its training.

During the preprocessing stage, images are read in from a folder, converted to grayscale, sharpened based on a filtering kernel, resized to be 256x256 pixels and normalized. Much of this can be modified based on what
the user prefers, however it should be noted that normalizing images MUST take place after sharpening, otherwise the image gets distorted beyond the user's control.

During the training stage, much of the attention should be towards the definition of our model, which can be manipulated and adjusted based on your needs. At the start of this stage the model is defined in all its layers, the top layer being the input layer, and the bottom layer being the output layer. It is worth noting that while many of the parameters for each layer are customizable and have no real set rule to what they 
should be set to, the final output layer should have the same number of "filters" as classifications we want the model to output. The input shape for the input layer should also be of 3 dimensions, where the first 
two parameters are the dimensions of an image in our dataset, and the third dimension is in relation to the colour code of the image (1 for gray scale, 3 for RGB, etc.). While training, the length of each epoch will
be determined by how large of a batch size we specify, so smaller batch sizes allow each epoch to be completed in a smaller amount of time, but at the expense of accuracy of our model, conversely with large batch 
sizes we get better accuracy, with longer training times.

During the prediction stage, we read in images from a folder and allow the model to tell us what classification label it should belong to based on its training. This file can also be adjusted to bin
each predicted image into a folder that holds others of the same classification, if the user so desires.

# Training Model Log

## 2023-02-10 Run 1:
Test Set Accuracy - 14/20
Speed - 1354s -> 22.57 mins.

## 2023-02-10 Run 2: 
Changed batch size from 64 to 32.

Test Set Accuracy - 12/20
Speed - 1198s -> 19.97 mins.

## 2023-02-10 Run 3:
Changed number of epochs from 55 to 65.

Test Set Accuracy - 12/20
Speed - 1234s -> 20.57 mins.
Training Loss at End -> 1.1559
Training Accuracy at End -> 0.6161

## 2023-02-16 Run 4:
Change Epochs from 65 back to 55.
Change Batch size from 32 to 64.
Change all Conv2D layer padding functions to valid.

Test Set Accuracy - 8/20
Speed - 986s -> 16.43 mins.
Training Loss at End -> 1.5678
Training Accuracy at End -> 0.4226

## 2023-02-16 Run 5:
Change input layer kernel size to 5.
Change input layer strides to 2.
Change all Conv2D layers to have padding "same".

Test Set Accuracy - 8/20
Speed - 263s -> 4.38 mins.
Training Loss at End -> 1.6987
Training Accuracy at End -> 0.3065

## 2023-02-16 Run 6:
Change Epochs from 55 to 100.

Test Set Accuracy - 7/20
Speed - 457s -> 7.62 mins.
Training Loss at End -> 1.4561
Training Accuracy at End -> 0.4419

## 2023-02-17 Run 7:
Change Epochs from 100 to 200.
Change optimizer learning rate from 0.00001 to 0.0001.

Test Set Accuracy - 16/20
Speed - 812s -> 13.53 mins.
Training Loss at End -> 0.0065
Training Accuracy at End -> 1.0000

## 2023-02-17 Run 8:
Change Epochs from 200 to 70

Test Set Accuracy - 13/20
Speed - 286s -> 4.77 mins.
Training Loss at End -> 0.4428
Training Accuracy at End -> 0.8774

## 2023-02-17 Run 9:
Change Epochs from 70 to 80

Test Set Accuracy - 13/20
Speed - 342s -> 5.70 mins.
Training Loss at End -> 0.2677
Training Accuracy at End -> 0.9452

## 2023-02-17 Run 10:
Change Epochs from 80 to 90

Test Set Accuracy - 14/20
Speed - 401s -> 6.68 mins.
Training Loss at End -> 0.2388
Training Accuracy at End -> 0.9710

## 2023-03-07 Run 11:
Changed learning rate of optimizer to 0.001

Test Set Accuracy - 15/20
Speed - 459s -> 7.65 mins.
Training Loss at End -> 1.1466e-04
Training Accuracy at End -> 1.0000

## 2023-03-07 Run 12:
Changing Number of epochs to 30

Test Set Accuracy - 16/20
Speed - 151s -> 2.5167 mins.
Training Loss at End -> 0.0446
Training Accuracy at End -> 0.9968