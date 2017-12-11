# split train set and validation set for single model testing

import os
import shutil
import numpy as np

path = '../data/'
if not os.path.exists(path+'image_train/'):
    os.mkdir(path+'image_train/')
if not os.path.exists(path+'image_val/'):
    os.mkdir(path+'image_val/')


filename_list = []
for d in os.listdir(path+'image/'):
    if not os.path.exists(path+'image_train/'+d):
        os.mkdir(path+'image_train/'+d)
    if not os.path.exists(path+'image_val/'+d):
        os.mkdir(path+'image_val/'+d)
    
    filename = []
    for file in os.listdir(path+'image/'+d):
        filename.append(d+'/'+file)
    filename_list.append(filename)

for i in range(len(filename_list)):
    np.random.shuffle(filename_list[i])

    
split_rate = 0.8
for i in range(len(filename_list)):
    for f in filename_list[i][:int(len(filename_list[i])*split_rate)]:
        shutil.copyfile(path+'image/'+f,path+'image_train/'+f)
    for f in filename_list[i][int(len(filename_list[i])*split_rate):]:
        shutil.copyfile(path+'image/'+f,path+'image_val/'+f)
        