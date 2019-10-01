import json
import os
import os.path
import shutil
import cv2
import logging
import datetime
from datetime import datetime, timedelta
from timer import *

input_folder_path = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/even light/'
output_folder_path= '/home/benjamin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/partition_testing/even light/'
path_i = input_folder_path + 'Andy_right&left_10lux/'
path_o = output_folder_path + 'Andy_right&left_10lux/'
l = os.listdir(path_i)
img_cnt = len(l)

while os.path.exists(path_o):
   shutil.rmtree(path_o)
for retry in range(1000):
    try:
        os.mkdir(path_o)
        break
    except:
        print( "mkdir failed, retrying...", retry )

for file_idx in range(1,img_cnt+1,1):
    print(file_idx, end = ' ')
    filename = path_i + 'Andy_right&left_10lux_'+ f"{file_idx:02d}" + '.pgm'
    # print(filename)
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    height, width = img.shape
    try:
        assert (width == 320 and height == 240)
    except AssertionError as err:
        logging.error(str(err) + f"w: {width}\th: {height}")
    # row, col = 0, 0
    # for i in range(3):
    #     for j in range(3):
    #         cv2.imwrite(path_o + 'Andy_right&left_10lux_' + f"{file_idx:02d}" + "_partition_" + str(3*i+j) + ".pgm", img[row:row+160][col:col+160])
    #         col += 80
    #     col = 0
    #     row += 40