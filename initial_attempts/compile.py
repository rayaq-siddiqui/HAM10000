# data science libraries
import os
import pandas as pd
# import cv2
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1000)

# keras and tf
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# other tf imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model imports
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

# for compiling the model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy as cce
from tensorflow.keras.metrics import CategoricalAccuracy as cca

# transfer learning models
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.utils import class_weight as cw

# print validation statement
print("all resources loaded")

# class_weight = {
#     0: 0.21338020666879728,
#     1: 1.2854575792581184,
#     2: 1.301832835044846,
#     3: 2.78349082823791,
#     4: 4.375273044997815,
#     5: 10.075452716297788,
#     6: 12.440993788819876
#  }

# class_weight = {
#     0: 1.302,
#     1: 0.213,
#     2: 12.441,
#     3: 1.285,
#     4: 10.075,
#     5: 2.783,
#     6: 4.375
# }

# class_weight = {
#     0: 0.2132609332162155,
#     1: 1.2903849251087132,
#     2: 1.3140888961784485,
#     3: 2.71869697997964,
#     4: 4.436323366555925,
#     5: 9.459268004722551,
#     6: 13.155993431855501
# }

df = None
X_train = np.load('variables/X_train_resized.npy')
X_test = np.load('variables/X_test_resized.npy')

y_train = np.load('variables/y_train.npy')
y_test = np.load('variables/y_test.npy')

yt = []
for y in (y_train):
    for i, num in enumerate(y):
        if num == 1:
            yt.append(i)
yt = list(yt)

class_weight = cw.compute_class_weight(
    class_weight='balanced',
    classes=range(7),
    y=yt
)

class_weight = dict(enumerate(class_weight))

print('all data loaded')

def seq_model(out=7):
    # sequential attempt
    inp =  Input((30, 40, 3))
    # x = BatchNormalization()(inp)

    x = Conv2D(128, kernel_size=5, activation='relu')(inp) #, kernel_initializer='glorot_normal')(inp)
    x = Conv2D(128, kernel_size=5, activation='relu')(x) #, kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(256, kernel_size=3, activation='relu')(x) #, kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, kernel_size=3, activation='relu')(x) #, kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    # x = Dropout(0.2)(x)

    # x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(x)
    # x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(x)
    # x = MaxPooling2D(pool_size=(2,2))(x)

    # x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(x)
    # x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(x)
    # # x = GlobalAveragePooling2D()(x)
    # x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.3)(x)
    out = Dense(out, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())
    return model


def seq_model2(out=7):
    # sequential attempt
    inp =  Input((30, 40, 3))
    x = tf.keras.layers.Normalization()(inp)

    x = Conv2D(256, kernel_size=5, activation=tf.keras.layers.PReLU())(x) #, kernel_initializer='glorot_normal')(inp)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, kernel_size=3, activation=tf.keras.layers.PReLU())(x) #, kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, kernel_size=2, activation=tf.keras.layers.PReLU())(x) #, kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(1024, activation=tf.keras.layers.PReLU())(x)
    x = Dense(128, activation=tf.keras.layers.PReLU())(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.3)(x)
    out = Dense(out, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())
    return model

def resnet(out=7):
    model = ResNet50(weights='imagenet',include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(out)(x)

    for layer in model.layers[0:30]:
        layer.trainable = False

    model_final = Model(model.input, out)

    print(model_final.summary())
    return model_final

model = None
model = resnet(7)

print('model created')

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
# optimizer = RMSprop(learning_rate = 1, rho=0.9, momentum=0.5, epsilon=1e-07)
# optimizer = SGD(lr = 0.001)

loss=cce()

acc = [cca()]
# METRICS = [
#       tf.keras.metrics.TruePositives(name='tp'),
#       tf.keras.metrics.FalsePositives(name='fp'),
#       tf.keras.metrics.TrueNegatives(name='tn'),
#       tf.keras.metrics.FalseNegatives(name='fn'),
#       tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#       tf.keras.metrics.Precision(name='precision'),
#       tf.keras.metrics.Recall(name='recall'),
#       tf.keras.metrics.AUC(name='auc'),
#       tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
# ]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=acc
)

print('model compiled')

lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=4, verbose=1,
    mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000000001
)

print('callback created')

model.fit(
    X_train, y_train,
    validation_split=0.2,
    # validation_data=(X_test, y_test),
    epochs=75,
    verbose=1,
    callbacks=[lr_reduction],
    batch_size=64,
    # class_weight=class_weight
)

print('model fit')

model.save('model')
