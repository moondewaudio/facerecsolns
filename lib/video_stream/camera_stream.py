#!/usr/bin/env python3
"""
camera_stream.py

Uses computer webcams to generate a stream.

Author: Simon Fong
"""

class CameraStream:
    def __init__(self,camera=0,video_file=None,video=False):
        from cv2 import VideoCapture

        if(video):
            # Use a video file as input.
            self.stream = VideoCapture(video_file)
        else:
            # Use a camera as input.
            self.stream = VideoCapture(camera)

        # Check if we were successful in opening stream.
        if(self.stream.isOpened() == False):
            name = video_file if video else camera
            raise IOError("Error opening video stream or file '{}'".format(name))

    def __del__(self):
        """ Destructor to close everything. """
        # When everything done, release the video capture object    
        self.stream.release()
		 

    def get_frame(self):
        """ Retrieve frame. """
        # Read until video is completed
        while(self.stream.isOpened()):
            # Capture frame-by-frame
            ret,frame = self.stream.read()
            if(ret == True):
                yield frame
		 
		


def _main(args):
    import cv2
    cs = CameraStream(args.camera,args.file,args.video)
    print("Press Q to on video window to exit.")
    for frame in cs.get_frame():
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    
if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument('-c','--camera',help='Specify camera to use.', type=int, default=0)
    video_group.add_argument('-f','--file',help='Specify video file.')
    parser.add_argument('-v','--video',help='Flag to use video file.',default=False,action='store_true')
    args = parser.parse_args()
    _main(args)

