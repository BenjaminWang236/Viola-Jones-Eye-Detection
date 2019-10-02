# import timeit
# code = """

# import json
import sys
import os
# import os.path
# import shutil
import cv2
import logging
import datetime
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import natsort
# import pprint as pp
from natsort import natsorted, ns
from datetime import datetime, timedelta

start = datetime.now()

# input_path = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/even light/'
# input_path = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/uneven light/5-100lux/'
input_path = '/home/benjamin/OneDrive/NeuronBasic/big_image_320_240/uneven light/100-1000lux/'
# output_path= '/home/benjamin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/partition_testing/even light/'
# output_path= '/home/benjamin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/partition_testing/uneven light/5-100lux/'
output_path= '/home/benjamin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/partition_testing/uneven light/100-1000lux/'

if not os.path.exists(input_path) and not os.path.exists(output_path):
    logging.error('INPUT | OUTPUT PATH DOES NOT EXIST')
    sys.exit(1)
o = [x[1] for x in os.walk(input_path)]
sublist = natsorted(o[0])

for subfolder in sublist:
    path_i = input_path + subfolder + '/'
    path_o = output_path + subfolder + '/'
    if not os.path.exists(path_i):
        break
    print(f'Subfolder {subfolder}')

    img_list = os.listdir(path_i)
    img_list = natsorted(img_list)
    img_cnt = len(img_list)
    i = 0
    while not os.path.exists(path_o) and i < 1000:
        try:
            os.mkdir(path_o)
            break
        except:
            # logging.warn("re: mkdir attempt %d" % i)
            if i == 999:
                print(f"re: mkdir attempt {i}")
            i += 1       

    for file_idx in range(1,img_cnt+1,1):
        print(file_idx, end = ' ')
        imagename = img_list[file_idx-1]
        filename = path_i + imagename
        print(filename)
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
                cv2.imwrite(path_o + imagename[:-4] + f'_partition_{i*3+j}.pgm', img[row:row+160, col:col+160])
                col += 80
            col = 0
            row += 40

end = datetime.now()
runtime = end-start
print(f'\nScript Runtime: {runtime}s')
# """

# runtime = timeit.timeit(code, number=100)/100
# print(runtime)