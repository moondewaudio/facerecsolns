

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grabreference
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

face_cascade = cv2.CascadeClassifier('/home/pi/opencv-2.4.13.4/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/opencv-2.4.13.4/data/haarcascades/haarcascade_eye.xml')
 

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    
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


    rawCapture.truncate(0)

    if key == ord("q"):
        break

