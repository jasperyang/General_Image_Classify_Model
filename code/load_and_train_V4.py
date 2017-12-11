# -*- encoding=utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from tensorflow.python.ops import math_ops

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


def load_model_and_finetune(MODEL,image_size,model_path,new_model_path,lambda_func=None):
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
    
    
    for layer in model.layers[:400]:
        layer.trainable = False
    for layer in model.layers[400:]:
        layer.trainable = False

    gen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

    gen2 = ImageDataGenerator(rescale=1./255)

    train_generator = gen.flow_from_directory(path+"image_train", image_size, shuffle=False,
                                              batch_size=batchsize,class_mode='categorical')
    validation_generator = gen2.flow_from_directory(path+"image_val", image_size, shuffle=False,
                                             batch_size=batchsize,class_mode='categorical')
    
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=[log_loss])

    model.fit_generator(
                train_generator,
                samples_per_epoch=train_generator.samples,
                nb_epoch=50,
                validation_data=validation_generator,
                nb_val_samples=validation_generator.samples/batchsize)


    model.save_weights(new_model_path)
    

load_model_and_finetune(InceptionResNetV2,(299,299),'../model/InceptionResNetV2_e50.h5','../model/incepV4_400.h5')