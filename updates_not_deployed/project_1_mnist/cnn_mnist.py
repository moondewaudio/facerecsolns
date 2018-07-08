"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from: https://keras.io/getting-started/functional-api-guide/

Modify this code to write a LeNet with the following requirements:
* Input dimensions: 32x32x1 - Input
* C1: Convolutional Layer - Conv2D
    number of filters: 6
    kernel size: 5x5
    strides: 1 both horizontally and vertically (Set by default.)
    activation function: sigmoid
    output: 6 layers of 28x28 feature maps (Do not need to specify in function.)
* S2: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 6 layers of 14x14 feature maps (Do not need to specify in function.)
* C3: Convolutional Layer - Conv2D
    number of filters: 16
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 16 layers of 10x10 feature maps(Do not need to specify in function.)
* S4: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 16 layers of 5x5 feature maps (Do not need to specify in function.)
* C5: Convolutional Layer - Conv2D
    number of filters: 120
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 120 layers of 1x1 feature maps(Do not need to specify in function.)
* F6: Fully Connected Layer - Dense
    units: 84
    activation function: tanh
    output 84-dimensional vector (This is specified through units.)
* F7: Fully Connected Layer - Dense
    units: 10
    activation function: softmax
    output 10-dimensional vector (This is specified through units.)
"""
# TODO: Import other layers as necessary. (Conv2D, MaxPooling2D)
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
import numpy as np
import cv2
import os

# Load MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Proccess the data from (28,28) to (32,32)
def procces_image(img):
	proccesed_image = cv2.resize(img, (32,32))
	return proccesed_image

x_train = np.array(map(procces_image, x_train))
x_test = np.array(map(procces_image, x_test))
print("Resized images to {}".format(x_train.shape))

x_train = np.reshape(x_train,(60000,32,32,1))
x_test = np.reshape(x_test,(10000,32,32,1))
print(x_train.shape)

# One hot encode labels.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = np.reshape(y_train,(60000,1,1,10))
y_test = np.reshape(y_test,(10000,1,1,10))
print(y_train.shape)


# TODO: Currently, sets input dimension to be 784x1. Change to 32x32x1
inputs = Input(shape=(32,32,1))

x = Conv2D(6,(5,5), activation='sigmoid')(inputs)   # Convolution
x = MaxPooling2D((2,2),strides=(2,2))(x)            # Pooling
x = Conv2D(16,(5,5), activation='sigmoid')(x)       # Convolution
x = MaxPooling2D((2,2),strides=(2,2))(x)            # Pooling
x = Conv2D(120,(5,5), activation='sigmoid')(x)      # Convolution
x = Dense(84, activation='tanh')(x)                 # Fully connected layer
predictions = Dense(10, activation='softmax')(x)    # Inference layer

# This creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

# Print model architecture
model.summary()

# Compile model
optimizer = keras.optimizers.SGD(lr=1e-4,momentum=0.9)
model.compile(optimizer,'categorical_crossentropy', metrics=['accuracy'])

# Setting for training.
NUM_EPOCHS = 20
BATCH_SIZE = 16

# Train the model.
model.fit(x=x_train,y=y_train,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS)

# Save the model.
model.save('yann_mnist.h5')

# Test the model.
metrics = model.evaluate(x=x_test,y=y_test, batch_size=BATCH_SIZE)

# Print out the accuracy.
print("{metrics_names[0]}: {metrics[0]} \n {metrics_names[1]}: {metrics[1]}".format(metrics=metrics, 
	metrics_names=model.metrics_names))



