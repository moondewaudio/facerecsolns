#!/usr/bin/env python3
"""
knn_sklearn.py

Trains a face recognizer.

Author: Simon Fong
"""
def set_path():
    """ Used to get the library path. """
    import os, sys
    path_of_file = os.path.abspath(os.path.dirname(__file__))
    repo_path = os.path.join(path_of_file,'../../lib')
    sys.path.append(repo_path)

# Setting path to find custom library.
set_path()
import cv2
import os
import numpy as np

def detect_face(img):
    """ From: https://github.com/informramiz/opencv-face-recognition-python"""
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def collect_faces(person,num_images):
    """ Captures faces images and stores them in a directory.
    """
    from video_stream.video_stream import VideoStream

    face_cascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')

    # Custom Video Stream library
    cam = VideoStream(picamera=False)

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

            cropped = resized[y:y+w,x:x+h]
            r_cropped = cv2.resize(cropped, (100, 100))
            # Save the image.
            cv2.imwrite(image_path, r_cropped)
            print("Face Detected {count:02}/{total:02}".format(count=count+1,total=num_images))
            count += 1
            
            # Draw the rectangle on the face.
            cv2.rectangle(resized,(x,y),(x+w,y+h),(255,255,255),2)

        
        # Show the image.
        cv2.imshow("Frame", resized)
        

        # If we captured enough images.
        if(count >= num_images):
            break

        # Wait one second and exit if 'q' is pressed.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(recognizer,test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if(rect is None):
        return test_img

    face = cv2.resize(face, (100, 100))

    #predict the image using our face recognizer 
    label, confidence = recognizer.predict(face)
    #get name of respective label returned by face recognizer
    #label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, str(label), rect[0], rect[1]-5)
    
    return img

def train(name, load=None, save=None):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()


    if(load is not None):
        face_recognizer.load(load)
        print("Loading model from {}".format(load))
        return face_recognizer

    images =[]
    labels = []
    index = 0

    image_paths = os.listdir(name)

    for image_path in image_paths:
        image_path = os.path.join(name,image_path)
        image  = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(gray)
        labels.append(index)

    face_recognizer.train(images, np.array(labels))

    if(save is not None):
        face_recognizer.save(save)
        print("Saved model to {}".format(save))

    return face_recognizer

def predict_live(recognizer):
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        predicted_image = predict(recognizer, frame)
        cv2.imshow('Face', predicted_image)
        # Wait one second and exit if 'q' is pressed.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

def _main(args):
    if(args.cap):
        collect_faces(args.person,args.count)

    if(args.train):
        recognizer = train(args.person, load=args.load, 
            save=args.save)

        if(args.predict):
            predict_live(recognizer)





  
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--person', help='Name to label the images.',default='simon')
    parser.add_argument('-c','--count', help='Number of images to collect.',default=10,type=int)
    parser.add_argument('-ca','--cap', help='Whether to capture images.',default=False,action='store_true')
    parser.add_argument('-t','--train', help='Whether to train',default=False,action='store_true')
    parser.add_argument('-pr','--predict', help='Whether to predict',default=False,action='store_true')
    parser.add_argument('-l','--load', help='Path to load model.',default=None)
    parser.add_argument('-s','--save', help='Path to save model.',default=None)
    args = parser.parse_args()
    _main(args)