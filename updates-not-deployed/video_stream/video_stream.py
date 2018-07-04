#!/usr/bin/env python3
"""
video_stream.py

Uses any camera to generate a stream.

Author: Simon Fong
"""

class VideoStream:
    def __init__(self,camera=0,video_file=None,video=False,picamera=False,width=640,height=480,rotation=180):
        if(picamera):
            from picamera_stream import PiCameraStream
            self.camera = PiCameraStream(width,height,rotation)
        else:
            from camera_stream import CameraStream
            self.camera = CameraStream(camera,video_file,video)

    def __del__(self):
        """ Destructor to close everything. """
        # When everything done, release the video capture object
        pass  

    def get_frame(self):
         """ Retrieve frame. """
         for frame in self.camera.get_frame():
            yield frame

def _main(args):
    import cv2
    vs = VideoStream(args.camera,args.file,args.video,
        picamera=not args.webcam, width=args.width,height=args.height,rotation=args.rotation)
    print("Press Q to on video window to exit.")
    for frame in vs.get_frame():
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()

    # OpenCV VideoStream arguments
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument('-c','--camera',help='Specify camera to use.', type=int, default=0)
    video_group.add_argument('-f','--file',help='Specify video file.')
    parser.add_argument('-v','--video',help='Flag to use video file.',default=False,action='store_true')

    # PiCamera arguments
    parser.add_argument('-cw','--width',help='Specify width.', type=int, default=640)
    parser.add_argument('-ch','--height',help='Specify height', type=int, default=480)
    parser.add_argument('-r','--rotation',help='Specify rotation.', type=int, default=180)

    # Flag to whether to use PiCamera
    parser.add_argument('-web','--webcam',help='Specify to use webcam.',default=False,action='store_true')
    
    args = parser.parse_args()
    _main(args)