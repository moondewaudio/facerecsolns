#!/env/bin/python
"""
video_copier.py

Copies a video input stream to an output stream.

Author: Simon Fong
"""

import numpy as np
import os
import sys
import cv2

class VideoCopier:
    """Class to load a video and classify each frame of a video"""
    
    def __init__(self, input_name, output_name=None, show_image=False):
        """Setup input video stream and output video stream """
        
        OUTPUT_PREFIX = 'output_'


        # Keeping track of file names.
        self.input_name = input_name
        self.output_name = output_name

        # If output name not specified.
        if(output_name is None):
            self.output_name = OUTPUT_PREFIX + input_name
        
        # Flag to determine to convert video file after writing
        self.CONVERT_TO_MP4 = False
        
        # Flag for whether or not to show images while processing
        self.SHOW_IMAGE = show_image
        
        # For progress spinner
        self.iterations = 0
        
    def create_video_input(self, input_name):
        """Define VideoCapture object"""
        if(input_name == '0'):
            # 0: Built in webcam
            self.input_video = cv2.VideoCapture(0)       
        elif(input_name == '1'):
            # 1: External webcam
            self.input_video = cv2.VideoCapture(1)       
        else:
             # If not webcam, the open video
            self.input_video = cv2.VideoCapture(input_video_name) 
    
    def create_video_writer(self, output_name):
        """Define VideoWriter object"""
        
        # Save output video name
        self.output_name = output_name.split('.')[0]
        self.extension = output_name.split('.')[-1]
        
        # If mp4 file, save as avi then convert
        if(self.extension == 'mp4'):
            self.output_video_name_temp = self.output_name + '.avi'
            self.CONVERT_TO_MP4 = True
        else:
            self.output_video_name_temp = '.'.join(self.output_name, self.extension)
        
        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*'X264')

        # Set FPS from video file
        fps = self.input_video.get(cv2.CAP_PROP_FPS)
        
        # Get videocapture's shape
        out_shape = (int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Create VideoWriter object
        self.output_video = cv2.VideoWriter(self.output_video_name_temp, 
            fourcc, fps, out_shape)
    
    
    def get_frame(self):
        """Takes a frame from the video and returns it in the proper format"""
        
        # Capture frame-by-frame
        ret, original_image = self.input_video.read()
        
        # Check if frame was captured
        if(ret == False):
            raise TypeError("Image failed to capture")

        # Creates an image list because the model expects 4 dimensions
        images = []

        # Resize image
        image_resized = cv2.resize(original_image, (self.dataset.height,
                            self.dataset.width))
        images.append(image_resized)
        
        # Convert list to numpy array
        images = np.array(images)

        return (images, original_image)
    
    def classify(self, input_video_name, output_video_name):
        """Classify all the frames of the video and save the labeled video"""
        
        # Create video capture object
        self.create_video_input(input_video_name)
        
        # Create video writer object
        self.create_video_writer(output_video_name)
        
        # Continuously grab a frame from the camera and classify it
        print("Classifying the video")
        while(self.input_video.isOpened()):
            self.spin()
            try:
                # Capture image from webcam
                images, original_image = self.get_frame()
            
            except TypeError, e:
                # Break if image failed to capture
                print(e)
                break
            
            # Classify image
            label = self.model.predict(images)
            label = self.dataset.names[np.argmax(label[0])].split('.')[1]
            
            # Print the text on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_image,label,(50,50), font, 
                        1,(255,255,255),2,cv2.LINE_AA)
            
            # Write image to video
            self.output_video.write(original_image)
                       
            # Show the image if flag is set
            if(self.SHOW_IMAGE):
                cv2.imshow('image',original_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        self.release()
        
        if(self.CONVERT_TO_MP4):
            self.convert_avi_to_mp4(self.output_video_name_temp,
                self.output_video_name)
        
        return self.output_video_name
        
    def release(self):
        """Release and destory everything"""
        
        self.input_video.release()
        self.output_video.release()
        cv2.destroyAllWindows()
        print("Finished processing video, saving file to {}".format(
            self.output_video_name))
        
    def convert_avi_to_mp4(self, avi_file_path, output_name):
        cmd = "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4' -y".format(
            input = avi_file_path, output = output_name)
        call(cmd,shell=True)
        os.remove(avi_file_path)
        return True
        
    def spin(self):
        "Spin the progress spinner"
        self.iterations += 1
        spin_states = {
                        0: "-",
                        1: "\\",
                        2: "|",
                        3: "/",
                        4: "-",
                        5: "\\",
                        6: "|",
                        7: "/",
                        }
        state = spin_states[self.iterations%8]
        sys.stdout.write(state + "\r")
        sys.stdout.flush()
                   
def main():

    # Get input video from command line
    input_video_name = str(sys.argv[1])
    
    # Get model from command line
    model_name = str(sys.argv[2])

    # Get output video name from command line
    output_video_name = str(sys.argv[3])

    # Get show image flag from command line
    show_image = sys.argv[4] == 'True'

    video_classifier = VideoClassifier(dataset_name, image_height,      
        image_width, model_name, show_image)
    
    try:
        video_classifier.classify(input_video_name, output_video_name)
    
    except KeyboardInterrupt:
        # Release and destory everything
        video_classifier.release()
        print("Exiting on Interrupt")
        
        

if __name__ == '__main__':
    main()
    
