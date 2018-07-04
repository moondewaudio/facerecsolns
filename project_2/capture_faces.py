
"""
NOTE: This should be in the student code, but is not necesaary because of the video_stream library.
from picamera.array import PiRGBArray
from picamera import PiCamera
"""
def set_path():
    """ Used to get the library path. """
    import os, sys
    path_of_file = os.path.abspath(os.path.dirname(__file__))
    repo_path = os.path.join(path_of_file,'../lib')
    sys.path.append(repo_path)

# Setting path to find custom library.
set_path()

import time
import cv2
import os
import sys
from video_stream.video_stream import VideoStream


#center is a tuple
##Ex: (100,100)
def drawRectangle(image,center):
    '''
    height,width = resized.shape

    top = (width/2-50, height/2-50)
    bottom = (width/2+50, height/2+50)

    cv2.rectangle(resized, top, bottom, (255,255,255),3)

   top = None
    bottom = None
    cv2.rectangle(image, top, bottom, (255,2555,255), 3)
    '''
    pass

"""
NOTE: This should be in the student code, but is not necesaary because of the video_stream library.
# initialize the camera and grabreference
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
"""

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')
 

time.sleep(0.1)

i = 0

person = str(sys.argv[1])
numImages = int(sys.argv[2])

"""
NOTE: This should be in the student code, but is not necesaary because of the video_stream library.
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
"""
# Custom Video Stream library
cam = VideoStream()

for image in cam.get_frame():
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    resized = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    
      
    faces = face_cascade.detectMultiScale(resized, 1.3, 5)
    

    for (x,y,w,h) in faces:
        dirname = person
        picName = os.path.join(dirname,person)+'_' + str(i).zfill(2) + '.jpg'

        if i < numImages:
            cv2.imwrite(picName, gray)
            print("FACE")
            i += 1
        else:
            exit()
        
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,255,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
    
   
    cv2.imshow("Frame", resized)
    key = cv2.waitKey(1) & 0xFF

    """
    NOTE: This should be in the student code, but is not necesaary because of the video_stream library.
    rawCapture.truncate(0)
    """

    if key == ord("q"):
        break

