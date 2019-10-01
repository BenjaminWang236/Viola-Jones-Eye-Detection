import json
import os
import os.path
import shutil
import cv2
import logging
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import natsort
import pprint as pp
from natsort import natsorted, ns
from datetime import datetime, timedelta
from timer import *

uneven_path_1 = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/uneven light/5-100lux/'
uneven_path_2 = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/uneven light/100-1000lux/'
even_path = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/even light/'
output_folder_path= '/home/benjamin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/partition_testing/even light/'
o = [x[1] for x in os.walk(even_path)]
sublist = natsorted(o[0])
pp.pprint(sublist)
pp.pprint(len(sublist))

path_i = even_path + 'Andy_right&left_10lux/'
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
    imagename = 'Andy_right&left_10lux_'+ f"{file_idx:02d}" + '.pgm'
    filename = path_i + imagename
    # print(filename)
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    try:
        assert (width == 320 and height == 240)
    except AssertionError as err:
        logging.error(str(err) + f"w: {width}\th: {height}")
    row, col = 0, 0
    for i in range(3):
        for j in range(3):
            cv2.imwrite(path_o + imagename[:-4] + "_partition_" + str(3*i+j) + ".pgm", img[row:row+160, col:col+160])
            # cv2.imshow('test', img[row:row+160][col:col+160])
            # cv2.waitKey()
            col += 80
        col = 0
        row += 40
print('\n')