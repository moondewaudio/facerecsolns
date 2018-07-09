#!/usr/bin/env python3
"""
svm_sklearn.py

Performs classification on the MNIST dataset using SVM.

Author: Simon Fong
"""

def _main(args):
    
    import numpy as np
    from sklearn.svm import SVC
    import time
    

    # Load dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))
    print("Reshaped x data into {}".format(x_train.shape))

    # Create KNN
    svm = SVC(cache_size=2048)


    start_time = time.time()
    # Train with data.
    print("Begin fitting...")
    svm.fit(x_train[:], y_train[:])
    print("Classifier fitted.")
    cur = time.time()
    fit_time = cur - start_time
    print("Fit time {} seconds".format(fit_time))

    # Do a test prediction.
    print("Begin predicting.")
    acc = svm.score(x_test, y_test)
    print("Accuracy: {}".format(acc))
    predict_time = time.time() - cur
    print("Prediction time {} seconds".format(predict_time))



  
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    _main(args)