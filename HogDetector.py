import cv2
import numpy as np
from skimage import feature
from skimage import exposure

# hog = cv2.HOGDescriptor()
im = cv2.imread('trainimg_0.bmp')
# h = hog.compute(im)

(H, hogImage) = feature.hog(im, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)
cv2.waitKey(0)