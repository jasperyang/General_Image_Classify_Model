# -*- encoding=utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from tensorflow.python.ops import math_ops

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

path = '../data/'
batchsize = 64

def load_model_and_getResult(MODEL,image_size,model_path,lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    
    base_model = MODEL(input_tensor=x, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(30, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 读取参数
    model.load_weights(model_path)
    
    gen2 = ImageDataGenerator(rescale=1./255)

    test_generator = gen2.flow_from_directory(path+"test_A", image_size, shuffle=False,
                                             batch_size=batchsize,class_mode='categorical')
    
    test = model.predict_generator(test_generator)
    return test

def getResult(output_path,y_pred):
    import os
    import pandas as pd
    
    path = './data/test_A/test'
    filename = []
    for file in os.listdir(path):
        filename.append(file)

    f = []
    for s in filename:
        for i in range(30):
            f.append(s.split('.')[0])

    index = [i for i in range(1,31)]*len(filename)

    y_pred = y_pred.clip(min=0.003, max=0.995)

    import numpy as np
    def softmax(x):
        """Compute the softmax of vector x."""
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    for y in range(len(y_pred)):
        y_pred[y] = softmax(y_pred[y])


    prob = [list(y_pred[j]) for j in range(len(y_pred))]
    probs = []
    for p in prob:
        probs.extend(p)

    pd.DataFrame({'f':f,'index':index,'probs':probs}).to_csv(output_path,header=None,index=None)    
    print(max(pd.read_csv(output_path,header=None).iloc[:,2]))

    
y_pred = load_model_and_getResult(InceptionResNetV2,(299,299),'../model/InceptionResNetV2_e50.h5')
getResult('../result/incepV4_e50.csv',y_pred)