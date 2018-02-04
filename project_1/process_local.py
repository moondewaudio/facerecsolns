"""
ECE196 Face Recognition Project
Author: W Chen

Prerequisite: You need to install OpenCV before running this code
The code here is an example of what you can write to print out 'Hello World!'
Now modify this code to process a local image and do the following:
1. read geisel-hr.jpg
2. convert color to gray scale
3. resize to half of its original dimensions
4. draw a box at the center the image with size 10x10
5. save image with a new name to local directory
All the above steps should be in one function called process_image()
"""
import cv2

# TODO: edit this function
def process_image():
	image = cv2.imread("geisel.jpg",0)
	cv2.imshow("geisel",image)
	cv2.waitKey(0)
	resized = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	cv2.imshow("geisel",resized)
	cv2.waitKey(0)
	height,width = resized.shape
	top = (width/2-50, height/2-50)
	bottom = (width/2+50, height/2+50)
	cv2.rectangle(resized, top, bottom,(255,255,255),3)
	cv2.imshow("geisel", resized)
	cv2.waitKey(0)
	cv2.imwrite("geisel-bw-rectangle.jpg", resized)
	return


def hello_world():
	print 'Hello World!'
	return


def main():
	#hello_world()
	process_image()
	return


if __name__ == '__main__':
    main()
