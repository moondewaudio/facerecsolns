#!/usr/bin/env python3
"""
visualizer.py

Visualize image data.

Author: Simon Fong
"""
import cv2

class Visualizer:
    def __init__(self, x, y):
        self.x = x      # Data
        self.y = y      # Target

    def visualize(self,index):
        img = self.x[index]
        label = str(self.y[index])
        cv2.imshow(label,img)
        cv2.waitKey(1)
        


def _main(args):
    import sys
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    v = Visualizer(x_train,y_train)

    for i in range(len(x_train)):
        sys.stdout.write("Count: {i:05}/{total}\r".format(i=i+1,total=len(x_train)))
        sys.stdout.flush()
        v.visualize(i)
    
    
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    _main(args)