#!/usr/bin/env python3
"""
svm_sklearn.py

Performs classification on the MNIST dataset using SVM.

Author: Simon Fong
"""

def _main(args):
    
    import numpy as np
    from sklearn.svm import SVC

    # Load dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))
    print("Reshaped x data into {}".format(x_train.shape))

    # Create KNN
    knn = SVC(cache_size=2048)


    # Train with data.
    print("Begin fitting...")
    knn.fit(x_train[:1000], y_train[:1000])
    print("Classifier fitted.")


    # Do a test prediction.
    print("Begin predicting.")
    acc = knn.score(x_test, y_test)
    print("Accuracy: {}".format(acc))


  
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    _main(args)