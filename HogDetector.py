import logging
import cv2
import numpy as np
import imutils
from imutils import paths
from skimage import feature
from skimage import exposure

path = 'C:/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/hog_testing/'
# hog = cv2.HOGDescriptor()
im = cv2.imread(path + 'trainimg_0_80x80.bmp')
imgShape = im.shape
print('Image Shape: {}'.format(imgShape))
# print(im[0][0][:])
im = cv2.resize(im, (40, 40))
# h = hog.compute(im)
cell_dim = (8, 8)
block_dim = (2, 2)
orientations = 9
precision = 15
(H, hogImage) = feature.hog(im, orientations=orientations, 
    pixels_per_cell=cell_dim, cells_per_block=block_dim, 
    transform_sqrt=True, block_norm="L1", 
    visualize=True, feature_vector=False)
Shape = H.shape
print('HOG vectors\'s shape: {}'.format(Shape))    # (3, 3, 2, 2, 9)
print('Total vectors: {}'.format(H.ravel().shape[0]))
cell_vectors = np.zeros([int(imgShape[0]/cell_dim[0]), int(imgShape[1]/cell_dim[1]), orientations])
cell_count = np.zeros_like(cell_vectors)
vector_shape = cell_vectors.shape
print('Concatenated Vector\'s shape: {}'.format(cell_vectors.shape))

# SUM VECTORS TO CORRESPONDING CELL & COUNT TO BE AVERAGED LATER
for a in range(Shape[0]):
    for b in range(Shape[1]):   # For each block of cells
        for c in range(Shape[2]):
            for d in range(Shape[3]):   # For each cell in block
                for e in range(Shape[4]):   # For each vector/histogram bin of cell
                    cell_vectors[a+c][b+d][e] += H[a][b][c][d][e]
                    cell_count[a+c][b+d][e] += 1
# Average the vectors that contribute more than once
cell_vectors = np.divide(cell_vectors, cell_count, out=np.zeros_like(cell_vectors), where=cell_count != 0)

# WRITE TO FILE & RE-FORMAT (Human-readable)
data = np.zeros((vector_shape[2], vector_shape[0], vector_shape[1]))
string = '{:.' + str(precision) + 'f}\t'
with open(path+'hog_data.txt', 'w+') as f:
    f.write('# Vector Shape: {}\n'.format(vector_shape))
    for k in range(vector_shape[2]):
        f.write('\n# Vector/Histogram Bin {0} (Angles {1} - {2})\n'.format(k, 20*k, 20*(k+1)))
        for i in range(vector_shape[0]):
            for j in range(vector_shape[1]):
                # Meaningful Precision up to about 60
                # data[k][i][j] = np.around(cell_vectors[i][j][k], precision)
                f.write(string.format((cell_vectors[i][j][k])))
            f.write('\n')
# data = np.around(data, precision)   # This is slower than rounding at each vector

# # READ FROM FILE (Human-readable)
# new_data = np.loadtxt(path+'hog_data.txt').reshape((vector_shape[2], vector_shape[0], vector_shape[1]))
# # new_data = np.around(new_data, precision)
# # assert np.all(new_data == cell_vectors)
# try:
#     assert np.all(new_data == data)
#     print("TRUE: Data read back is equal to original data")
# except AssertionError as error:
#     logging.error(" Data read back is not equal to original data")
#     diff = np.zeros_like(new_data)
#     for k in range(vector_shape[2]):
#         for i in range(vector_shape[0]):
#             for j in range(vector_shape[1]):
#                 diff[k][i][j] = data[k][i][j] - new_data[k][i][j]
#     print(diff)

    

# for k in range(Shape[4]):
#     fname = path + 'vec_bin' + str(k) + '.txt'
#     with open(fname, "a+") as outfile:
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
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imwrite(path + "hog_0.bmp", hogImage)

