import cv2
from sys import argv

img_path = argv[1]

img = cv2.imread(img_path)
cv2.imshow('',img)
cv2.waitKey(0)
