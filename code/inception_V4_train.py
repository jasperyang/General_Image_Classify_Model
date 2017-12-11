# -*- encoding=utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from tensorflow.python.ops import math_ops
from keras.applications.inception_resnet_v2 import preprocess_input

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

path = '../data/'

import h5py

batchsize = 64
image_size = (299,299)
lambda_func = preprocess_input

width = image_size[0]
height = image_size[1]
input_tensor = Input((height, width, 3))
x = input_tensor
if lambda_func:
    x = Lambda(lambda_func)(x)

base_model = InceptionResNetV2(input_tensor=x, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(30, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# 先fine-tune全连接层
for layer in base_model.layers[:]:
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


# train = model.predict_generator(train_generator,100)
# test = model.predict_generator(test_generator)

# log_loss
def log_loss(y_true, y_pred):
    epsilon = 1e-7
    predictions = math_ops.to_float(y_pred)
    labels = math_ops.to_float(y_true)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    return -(math_ops.reduce_sum(math_ops.multiply(labels,math_ops.log(predictions)+epsilon)))/batchsize

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=[log_loss])

model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.samples,
            nb_epoch=50,
            validation_data=validation_generator,
            nb_val_samples=validation_generator.samples/batchsize)


# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:-10]:
    layer.trainable = False
for layer in model.layers[-10:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=[log_loss])

model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.samples,
            nb_epoch=50,
            validation_data=validation_generator,
            nb_val_samples=validation_generator.samples/batchsize)


model.save_weights('../model/InceptionResNetV2_e50.h5')