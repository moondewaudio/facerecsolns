"""
process_realtime.py

Detects faces in realtime and displays them.

Author: Simon Fong
"""
def set_path():
    """ Used to get the library path. """
    import os, sys
    path_of_file = os.path.abspath(os.path.dirname(__file__))
    repo_path = os.path.join(path_of_file,'../lib')
    sys.path.append(repo_path)

# Setting path to find custom library.
set_path()

import cv2
from video_stream.video_stream import VideoStream

def _main(args):
    # Initialize the camera and grabreference
    camera = VideoStream(picamera=not args.webcam)

    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

    for image in camera.get_frame():
        
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        
        resized = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        
          
        faces = face_cascade.detectMultiScale(resized, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(resized,(x,y),(x+w,y+h),(255,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        
       
        cv2.imshow("Frame", resized)
        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()

    # Flag to whether to use PiCamera
    parser.add_argument('-web','--webcam',help='Specify to use webcam.',default=False,action='store_true')

    args = parser.parse_args()
    _main(args)
