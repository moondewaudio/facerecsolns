#!/usr/bin/env python3
"""
knn_sklearn.py

Performs K nearest neighbors on some given data. Uses sklearn's KNN.

Author: Simon Fong
"""

def _main(args):
    
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    # Load dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))
    print("Reshaped x data into {}".format(x_train.shape))

    # Create KNN
    knn = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)


    # Train with data.
    print("Begin fitting...")
    knn.fit(x_train, y_train)
    print("Classifier fitted.")

    # Evaluate on testing data.
    print("Begin predicting.")
    acc = knn.score(x_test,y_test)
    print("Accuracy: {}".format(acc))


    # Do a test prediction.
    """
    tests = 10
    for i in range(tests):   
        x_predict = x_test[i]
        y_predict = knn.predict([x_predict])[0]
        y_correct = y_test[i]
        correct = False
        if(y_predict == y_correct):
            correct = True
        message = "Predicted: {predict}, Truth: {truth}, {}".format(correct,
            predict=y_predict,
            truth=y_correct)
        print(message)
    """

  
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    _main(args)