"""
predict_cnn_mnist.py

Predict MNIST images using a trained neural network.

ECE196 Face Recognition Project
Author: Simon Fong
"""
# TODO: Import other layers as necessary. (Conv2D, MaxPooling2D)
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
import numpy as np
import cv2

# Proccess the data from (28,28) to (32,32)
def procces_image(img):
    proccesed_image = cv2.resize(img, (32,32))
    return proccesed_image

def _main(args):
    # Load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Resize the images
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

    # Grab the first 10 images and labels
    images = x_test[:10]
    labels = y_test[:10]

    # Load the model
    model = load_model('yann_mnist.h5')

    # Predict and show each image
    for image,label in zip(images,labels):
        # Expands shape from (32,32,1) to (1,32,32,1)
        expanded_image = np.expand_dims(image,axis=0)

        # Uses model to predict
        predictions = model.predict(expanded_image)[0]
        
        # Find the index of largest value to get digit
        predicted_label = np.argmax(predictions)
        truth_label = np.argmax(label)
        
        # Show the image
        title = "Predicted: {}, Truth: {}".format(predicted_label, truth_label)
        print(title)
        cv2.imshow(title,image)
        cv2.waitKey()
        cv2.destroyAllWindows()


if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    _main(args)




