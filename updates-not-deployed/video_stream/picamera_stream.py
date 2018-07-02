#!/usr/bin/env python3
"""
picamera_stream.py

Uses computer webcams to generate a stream.

Author: Simon Fong
"""

class PiCameraStream:
    def __init__(self,width=640,height=480,rotation=180):
        from picamera import PiCamera
        from picamera.array import PiRGBArray

        # Init PiCamera
        self.stream = PiCamera()

        # Set resolution
        self.width = width
        self.height = height
        self.stream.resolution = (self.width,self.height)

        # Set rotation
        self.stream.rotation = rotation

        # Use for video stream
        self.raw_capture = PiRGBArray(self.stream, size=(self.width, self.height))

    def __del__(self):
        """ Destructor to close everything. """
        # When everything done, release the video capture object    
        pass

    def get_frame(self):
        """ Retrieve frame. """
        # Read until video is completed
        for f in camera.capture_continuous(self.raw_capture, format='bgr', use_video_port=True):
            frame = f.array
            yield frame

def _main(args):
    import cv2
    pcs = PiCameraStream(args.width,args.height,args.rotation)
    print("Press Q to on video window to exit.")
    for frame in pcs.get_frame():
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    
    
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cw','--width',help='Specify width.', type=int, default=640)
    parser.add_argument('-ch','--height',help='Specify height', type=int, default=480)
    parser.add_argument('-r','--rotation',help='Specify rotation.', type=int, default=180)
    args = parser.parse_args()
    _main(args)