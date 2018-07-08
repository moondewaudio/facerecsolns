#!/usr/bin/env python3
"""
knn.py

Performs K nearest neighbors on some given data.

Author: Simon Fong
"""

class KNN:
    def __init__(self, k=1):
        pass

    def __del__(self):
        """ Destructor to close everything. """
        pass

    def train(self, x_train, y_train):
        """ Trains the KNN with data.
        :x_train list(list(float)): Expects a 2-D list with each index corresponding to y.
        :y_train list: Expects a 2-D list with each index corresponding to x.
        """

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_predict):
        """ Predicts the label of the given data point.
        :x list: The datapoint.
        :return  type(y): The prediction.
        """

        # Assumes non-negative distances.
        min_distance = None

        # Track index
        min_index = None

        # Calculate the distance to each point in dataset.
        for idx, x in enumerate(self.x_train):
            # Calculate distance
            dist = self.distance(x_predict, x)

            # Handle when mutiple points are equidistant.
            if(min_distance == dist):
                # TODO
                pass

            # Init min distance.
            if(min_distance is None):
                min_distance = dist
                min_index = idx

            # Check if distance is less.
            if(dist < min_distance):
                min_distance = dist
                min_index = idx
            

        # Return the label of the nearest neighbor.
        return self.y_train[min_index]


    def distance(self, x_1, x_2):
        """ Calculates the distance from two data points.
        
        :x_1 list(floats): First datapoint.
        :x_2 list(floats): Second datapoint.
        :return float: Distance between points.
        """

        # Initialize distance
        distance = 0

        for (x_1_e, x_2_e) in zip(x_1, x_2):
            distance += abs(x_2_e - x_1_e)
        return distance

def _main_iris(args):
    # Load dataset
    from sklearn.datasets import load_iris
    data = load_iris()

    # Convert from numpy arrays to lists.
    x_train = [ list(x) for x in list(data.data)]
    y_train = list(data.target)

    # Convert from numpy.float to float.
    x_train = [[float(e) for e in x] for x in x_train]

    # Create KNN
    knn = KNN()

    # Train with data.
    knn.train(x_train,y_train)

    # Do a test prediction.
    x_predict = [5,5,5,5]
    y_predict = knn.predict(x_predict)
    print(y_predict)
    print(y_predict)

def _main_mnist(args):
    import cv2
    import numpy as np

    # Load dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Vectorize and to float data from np format
    x_train_v = vectorize(x_train)
    x_train_v = to_float(x_train_v)
    print("x train data formatted.")

    x_test_v = vectorize(x_test)
    x_test_v = to_float(x_test_v)
    print("x train data formatted.")

    y_train_v = to_float(y_train,label=True)
    print("y data formatted.")

    #cv2.imshow('Orignal',x_train[0])
    #cv2.imshow('Shaped',unshaped_x[0])
    #cv2.waitKey()

    # Create KNN
    knn = KNN()

    # Train with data.
    knn.train(x_train_v,y_train_v)

    # Do a test prediction.
    print("Begin predicting.")
    tests = 10
    for i in range(tests):   
        x_predict = x_test_v[i]
        y_predict = knn.predict(x_predict)
        y_correct = float(y_test[i])
        correct = False
        if(y_predict == y_correct):
            correct = True
        message = "Predicted: {predict}, Truth: {truth}, {}".format(correct,
            predict=y_predict,
            truth=y_correct)
        print(message)

def vectorize(data,single=False):
    import numpy as np

    # Handle all data
    if not single:
        (num_images,rows,cols) = data.shape
        return np.reshape(data,(num_images,rows*cols))
    
    # Handle a single image
    (rows,cols) = data.shape
    return np.reshape(data,(rows*cols))

def devectorize(data,shape,single=False):
    import numpy as np

    # Handle all data
    if not single:
        (num_images,num_elem) = data.shape
        return np.reshape(data,(num_images,shape[0],shape[1]))
    
    # Handle a single image
    (rows,cols) = data.shape
    return np.reshape(data,(rows*cols))

def to_float(data,label=False):
    # Handle x data
    if not label:
        # Convert from numpy arrays to lists.
        x_train = [ list(x) for x in list(data)]
        # Convert from numpy.float to float.
        x_train = [[float(e) for e in x] for x in x_train]
        return x_train

    # Handle labels
    y_train = list(data)
    return y_train

def _main(args):
    _main_mnist(args)

    


    
    
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    _main(args)