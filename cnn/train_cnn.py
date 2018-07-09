"""
train_cnn.py

Train cnn on MNIST data.

ECE196 Face Recognition Project
Author: Will Chen, Simon Fong

"""
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
import numpy as np
import cv2

# Proccess the data from (28,28) to (32,32)
def procces_image(img):
    proccesed_image = cv2.resize(img, (32,32))
    return proccesed_image

# Process and shape datasets
def get_dataset():
    # Load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Resize the images by applying the function to all the images.
    x_train = np.array(map(procces_image, x_train))
    x_test = np.array(map(procces_image, x_test))

    # Reshape to fit model
    x_train = np.reshape(x_train,(60000,32,32,1))
    x_test = np.reshape(x_test,(10000,32,32,1))
    print("Resized images to {}".format(x_train.shape))

    # One hot encode labels.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Reshape to fit model
    y_train = np.reshape(y_train,(60000,1,1,10))
    y_test = np.reshape(y_test,(10000,1,1,10))

    return (x_train, y_train), (x_test, y_test)

# Create the model.
def get_model():
    # TODO: Copy code from create_cnn.py
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

    return model

def _main(args):
    (x_train, y_train), (x_test, y_test) = get_dataset()

    model = get_model()
    

    # Compile model

    optimizer = keras.optimizers.SGD(lr=1e-4,momentum=0.9)

    # TODO: Compile model with the optimizer above, 'catergorical_crossentropy', and metrics=['accuracy']
    # Hint: model.compile
    model.compile(optimizer,'categorical_crossentropy', metrics=['accuracy'])

    # Setting for training.
    NUM_EPOCHS = 40         # NOTE: Change this to 1 while debugging it will run a lot faster.
    BATCH_SIZE = 16

    # TODO: Train the model.
    # Hint: Use model.fit
    model.fit(x=x_train,y=y_train,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS)

    # TODO: Save the model to 'yann_mnist.h5'
    # Hint: Use model.save
    model.save('yann_mnist.h5')


if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    _main(args)
