import json
import os
import os.path
import shutil
import cv2
import datetime
from datetime import datetime, timedelta
from timer import *

path = 'C:/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/'
path_i = path + 'trainimg/'
path_o = path + 'rescaled/'
list = os.listdir(path_i)
img_cnt = len(list)

while os.path.exists(path_o):
   shutil.rmtree(path_o)
for retry in range(1000):
    try:
        os.mkdir(path_o)
        break
    except:
        print( "mkdir failed, retrying...", retry )

height = input("Height to rescale to?\t")
width = input("Width to rescale to?\t")
dim = (int(width), int (height))

for file_idx in range(0,img_cnt,1):
    print(file_idx, end = ' ')
    filename = path_i + 'trainimg_'+ str(file_idx) + '.bmp'
    img = cv2.imread(filename)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
    cv2.imwrite(path_o+'trainimg_'+str(file_idx)+'.bmp',img)