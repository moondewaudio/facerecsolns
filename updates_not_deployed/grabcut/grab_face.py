#!/usr/bin/env python2
"""
grab_face.py

Extract foreground from background

Code from: https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class GrabFace:
    
    def __init__(self):
        pass
        
    def cut(self,image_path):
        img = cv.imread(image_path)
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (50,50,450,290)
        cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        plt.imshow(img),plt.colorbar(),plt.show()
        
        

def _main():
    from sys import argv
    if(len(argv) < 2):
        print("Not enough arguments.")
        return
    
    image_path = argv[1]
    gf = GrabFace()
    
    gf.cut(image_path)
    
    
if(__name__ == '__main__'):
    _main()
