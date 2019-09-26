# import timeit

# code = """
import cv2
import numpy as np
import imutils
from imutils import paths
from skimage import feature
from skimage import exposure

# hog = cv2.HOGDescriptor()
im = cv2.imread('trainimg_0.bmp')
# im = cv2.resize(im, (8, 8))
# h = hog.compute(im)

(H, hogImage) = feature.hog(im, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", 
    visualize=True, feature_vector=False)
print(H.shape)
# print(H)
H_rav = H[0].ravel()
print(H_rav.shape)
# np.savetxt('hog_1.txt', H)
# print(len(H[0]))
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")
# cv2.imwrite("hog_0.bmp", hogImage)
# """

# print(timeit.timeit(code, number=100)/100)

