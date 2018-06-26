"""
ECE 196 Face Recognition Project
Author: Will Chen, Simon Fong

What this script should do:
1. Start running the camera.
2. Detect a face, display it, and get confirmation from user.
3. Send it for classification and fetch result.
4. Show result on face display.
"""
import time,cv2, base64, requests
from picamera import PiCamera
from picamera.array import PiRGBArray

# Font that will be written on the image
FONT = cv2.FONT_HERSHEY_SIMPLEX

# TODO: declare useful paths here if you plan to use them
CASCADE_PATH = "/home/pi/opencv-2.4.13.4/data/haarcascades/haarcascade_frontalface_default.xml"
    
def request_from_server(img):
    """ 
    Sends image to server for classification.
    
    :param img: Image array to be classified.
    :returns: Returns a dictionary containing label and cofidence.
    """
    # URL or PUBLIC DNS to your server
    URL = "http://192.168.43.113:8080/predict"

    # File name so that it can be temporarily stored.
    temp_image_name = 'temp.jpg'

    # TODO: Save image with name stored in 'temp_image_name'
    cv2.imwrite(temp_image_name,img)

    # Reopen image and encode in base64
    image = open(temp_image_name, 'rb') #open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
     
    # Defining a params dict for the parameters to be sent to the API
    payload = {'image':image_64_encode}
     
    # Sending post request and saving the response as response object
    response = requests.post(url = URL, json = payload)
     
    # Get prediction from response
    prediction = response.json()

    return prediction


def main():
    # 1. start running the camera.
    # TODO: initialize face detector
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    # TODO: initialize camera and update parameters
    camera = PiCamera()
    WIDTH = 640
    HEIGHT = 480
    camera.rotation = 180
    camera.resolution = (WIDTH, HEIGHT)
    rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

    # Warm up camera
    print 'Let me get ready ... 2 seconds ...'
    time.sleep(2)
    print 'Starting ...'

    # 2. detect a face, display it, and get confirmation from user.
    for f in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        
        # Get image array from frame
        frame = f.array
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: use face detector to get faces
        # Be sure to save the faces in a variable called 'faces'
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            print '=================================='
            print 'Face detected!'
            cv2.imshow('Face Image for Classification', frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            cv2.waitKey()
            answer = input('Confirm image (1-yes / 0-no): ')
            print '=================================='

            if answer == 1:
                print 'Let\'s see who you are...'
                # TODO: get new result path and get name and confidence
                data = request_from_server(img)
                result_to_display = data['label']
                conf = data['confidence']
                
                print 'New result found!'

                # TODO: Display label on face image
                # Save what you want to write on image to 'result_to_display'
                # [OPTIONAL]: At this point you only have a number to display, 
                # you could add some extra code to convert your number to a 
                # name

                cv2.putText(frame, result_to_display, (10, 30), FONT, 1, (0, 255, 0), 2)
                cv2.imshow('Face Image for Classification', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey()
                break

        # Delete image in variable so we can get the next frame
        rawCapture.truncate(0)
        print 'Waiting for image...'
        time.sleep(1)
    return

# Runs main if this file is run directly
if(__name__ == '__main__'):
    main()
