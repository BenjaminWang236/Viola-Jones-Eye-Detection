# path = 'C:/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/hog_testing/'
path = '/home/benjamin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/hog_testing/'
# hog = cv2.HOGDescriptor()
im = cv2.imread(path + 'trainimg_0_80x80.bmp')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
