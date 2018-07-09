"""
capture_faces.py

Captures face images and stores them.

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
import os
from video_stream.video_stream import VideoStream

def _main(args):
    person = args.person
    numImages = args.count

    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

    # Custom Video Stream library
    cam = VideoStream(picamera=not args.webcam)

    # Make the directory to store the images if it doesn't exist.
    if not os.path.exists(person):
        os.mkdir(person)

    # Track how many images recorded.
    count = 0

    # Get images from the camera.
    for image in cam.get_frame():

        # Convert to grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        
        # Resize by 0.25
        resized = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        
        # Detect faces  
        faces = face_cascade.detectMultiScale(resized, 1.3, 5)
        
        # Draw a box on each face
        for (x,y,w,h) in faces:
            # Format image name
            dirname = person
            image_name = "{person}_{count:02}.jpg".format(person=person,count=count)
            image_path = os.path.join(dirname,image_name)

            # Save the image.
            cv2.imwrite(image_path, resized[x:x+h,y:y+w])
            print("Face Detected {count:02}/{total:02}".format(count=count+1,total=numImages))
            count += 1
            
            # Draw the rectangle on the face.
            cv2.rectangle(resized,(x,y),(x+w,y+h),(255,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Draw rectangle on the eyes.
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        
        # Show the image.
        cv2.imshow("Frame", resized)
        

        # If we captured enough images.
        if(count >= numImages):
            break

        # Wait one second and exit if 'q' is pressed.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()

    # Flag to whether to use PiCamera
    parser.add_argument('-web','--webcam',help='Specify to use webcam.',default=False,action='store_true')
    parser.add_argument('-p','--person', help='Name to label the images.',default='simon')
    parser.add_argument('-c','--count', help='Number of images to collect.',default=10,type=int)

    args = parser.parse_args()
    _main(args)
