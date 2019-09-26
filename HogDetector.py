import cv2
import numpy as np
from skimage import feature

hog = cv2.HOGDescriptor()
im = cv2.imread('trainimg_0.bmp')
