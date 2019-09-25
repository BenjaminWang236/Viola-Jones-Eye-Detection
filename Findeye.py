import json
import os
import os.path
import shutil
import cv2
import datetime
from datetime import datetime, timedelta
from timer import *

start = datetime.now()

###########################################################################
# Read in images and find eyes in them using openCV's cascade classifier
# and save result with eyes marked to trainimg1
###########################################################################

# Read the input image
#img = cv2.imread('C:/PythonApplication1/PythonApplication1/test.jpg')
#img = cv2.imread('C:/PythonApplication1/PythonApplication1/img_1.bmp')

# path = 'C:/CPP/ViolaJones/'
path = 'C:/Users/infin/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/'


path_i = path + 'trainimg/'
path_o = path + 'trainimg1/'
list = os.listdir(path_i)
img_cnt = len(list)
eyetable = []

while os.path.exists(path_o):
   shutil.rmtree(path_o)

# os.mkdir(path_o)
for retry in range(1000):
    try:
        os.mkdir(path_o)
        break
    except:
        print( "mkdir failed, retrying...", retry )

for file_idx in range(0,img_cnt,1):

    filename = path_i + 'trainimg_'+ str(file_idx) + '.bmp'
    img = cv2.imread(filename)

    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #face_cascade = cv2.CascadeClassifier('C:/PythonApplication1/PythonApplication1/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(img, 1.01, 1)

    i = 0
    if len(eyes) > 0 and len(eyes) < 3:
        for (x, y, w, h) in eyes:
            if x>1 and y>1 and (x+w)<63 and (y+h)<40 and w<32 and h<32:
                if i == 0:
                    xe0=x+w
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    eyetable.append([file_idx, int(x/2), int(y/2), int((x+w)/2), int((y+h)/2)])
                elif i == 1 and x >= xe0:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    eyetable.append([file_idx, int(x/2), int(y/2), int((x+w)/2), int((y+h)/2)])
                i = i + 1

        if i>0:
            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(path_o+'trainimg_'+str(file_idx)+'.bmp',img)




with open(path_o + 'eye_point_data.txt', 'w') as eye_point:
        eye_point.write(json.dumps(eyetable))

path_o = path_o[:-1]

nouse = input('Please review images in ' + path_o + ', then press ENTER key..........')

###########################################################################
# Based on manual evaluation of eye-marked images, retrieve the originals
# without the deleted images that didn't pass inspection
###########################################################################

path_exist = path + 'trainimg1/'
path_source = path + 'trainimg/'
path_copy = path + 'trainimg2/'

list = os.listdir(path_exist)
img_cnt = len(list)
trainimg = path_exist + 'trainimg_'+ str(img_cnt) + '.bmp'

while os.path.exists(trainimg):
    img_cnt = img_cnt + 1
    trainimg = path_exist + 'trainimg_'+ str(img_cnt) + '.bmp'

while os.path.exists(path_copy):
   shutil.rmtree(path_copy)

# os.mkdir(path_copy)
for retry in range(1000):
    try:
        os.mkdir(path_copy)
        break
    except:
        print( "mkdir failed, retrying...", retry )

for file_idx in range(0,img_cnt,1):
    trainimg = path_exist + 'trainimg_'+ str(file_idx) + '.bmp'

#    with open(trainimg) as f:
    try:
        f = open(trainimg)
        f.close()
        source = path_source + 'trainimg_'+ str(file_idx) + '.bmp'
        target = path_copy + 'trainimg_'+ str(file_idx) + '.bmp'
        shutil.copyfile(source, target)
        #os.remove(trainimg)
        #os.system('copy ' + source + target)
    except FileNotFoundError:
        no_use = []

###########################################################################
# Renaming to consecutive order and save to new folder for future use
###########################################################################

source_path = path + 'trainimg2/'
target_path = path + 'trainimg3/'

list = os.listdir(source_path)
img_cnt = len(list)

source = source_path + 'trainimg_'+ str(img_cnt) + '.bmp'
while os.path.exists(source):
    img_cnt = img_cnt + 1
    source = source_path + 'trainimg_'+ str(img_cnt) + '.bmp'

while os.path.exists(target_path):
   shutil.rmtree(target_path)

# os.mkdir(target_path)
for retry in range(1000):
    try:
        os.mkdir(target_path)
        break
    except:
        print( "mkdir failed, retrying...", retry )


i = 0
for file_idx in range(0,img_cnt,1):
    source = source_path + 'trainimg_'+ str(file_idx) + '.bmp'

#    with open(trainimg) as f:
    try:
        f = open(source)
        f.close()
        target = target_path + 'trainimg_'+ str(i) + '.bmp'
        shutil.copyfile(source, target)
        i = i + 1

        #os.remove(source)
    except FileNotFoundError:
        no_use = []


###########################################################################
# Perform the find eye operation again after aboves sorted unwanteds
###########################################################################


path_i = path + 'trainimg3/'
path_o = path + 'trainimg1/'
list = os.listdir(path_i)
img_cnt = len(list)
eyetable = []

while os.path.exists(path_o):
   shutil.rmtree(path_o)

# os.mkdir(path_o)
for retry in range(1000):
    try:
        os.mkdir(path_o)
        break
    except:
        print( "mkdir failed, retrying...", retry)

for file_idx in range(0,img_cnt,1):

    filename = path_i + 'trainimg_'+ str(file_idx) + '.bmp'
    img = cv2.imread(filename)

    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #face_cascade = cv2.CascadeClassifier('C:/PythonApplication1/PythonApplication1/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:/Ben Wang/OneDrive/NeuronBasic/Viola-Jones-Eye-Detection/haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(img, 1.01, 1)

    i = 0
    if len(eyes) > 0 and len(eyes) < 3:
        for (x, y, w, h) in eyes:
            if x>1 and y>1 and (x+w)<63 and (y+h)<40 and w<32 and h<32:
                if i == 0:
                    xe0=x+w
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    eyetable.append([file_idx, int(x/2), int(y/2), int((x+w)/2), int((y+h)/2)])
                elif i == 1 and x >= xe0:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    eyetable.append([file_idx, int(x/2), int(y/2), int((x+w)/2), int((y+h)/2)])
                i = i + 1

        if i>0:
            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(path_o+'trainimg_'+str(file_idx)+'.bmp',img)




with open(path_o + 'eye_point_data.txt', 'w') as eye_point:
        eye_point.write(json.dumps(eyetable))

path_o = path_o[:-1]




""" Timing how long it took to execute in total """
duration = datetime.now() - start
print('\n%s Total Duration %s %s' %
      ('-'*5, strfdelta(duration, '%H:%M:%S.%F'), '-'*5))
