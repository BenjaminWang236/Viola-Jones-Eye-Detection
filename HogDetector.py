# import timeit

# code = """
import cv2
import numpy as np
import imutils
from imutils import paths
from skimage import feature
from skimage import exposure

path = 'C:/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/hog_testing/'
# hog = cv2.HOGDescriptor()
im = cv2.imread(path + 'trainimg_0.bmp')
# im = cv2.resize(im, (8, 8))
# h = hog.compute(im)

(H, hogImage) = feature.hog(im, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1", 
    visualize=True, feature_vector=False)
Shape = H.shape
print(Shape)

# for k in range(Shape[4]):
#     fname = path + 'vec_bin' + str(k) + '.txt'
#     with open(fname, "w+") as outfile:
#         for i in range(Shape[0]):
#             for j in range(Shape[1]):
#                 outfile.write(str(H[i][j][0][0][k]) + '\n')

# # Below snippet confirms that cells start at top-left, 
# # move right until end then go to next row
# idx = np.ravel_multi_index([[0], [1], [0], [0], [0]], Shape)
# print(idx)
# orig_idx = np.unravel_index(idx, Shape)
# print(orig_idx)

# H_rav = H.ravel()
# np.savetxt(path + 'hog_test.txt', H_rav)
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")
# cv2.imwrite(path + "hog_0.bmp", hogImage)
# """

# print(timeit.timeit(code, number=100)/100)

