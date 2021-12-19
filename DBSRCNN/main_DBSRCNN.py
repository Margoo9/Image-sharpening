import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset.dataset_handling import load_data, deprocess_image
from model import generator_model, discriminator_model, generator_containing_discriminator, perceptual_loss, wasserstein_loss



data = load_data('./dataset/Train')
train_Y, train_X = data['sharp'], data['blur']

data_test = load_data('./dataset/Test')
test_Y, test_X = data['sharp'], data['blur']

# labels_num = len(np.unique(train_Y))

# normalize labels
# class_totals = train_Y.sum(axis=0)
# class_weight = class_totals.max() / class_totals


EPOCHS_NUM = 30
INIT_LEARNING_RATE = 1e-3
BATCH_SIZE = 64


model = Sequential()
channel_dim = -1

model.add(Conv2D(32, (9, 9), padding="same", activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(32, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(1, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(labels_num, activation='softmax'))

optim=Adam(lr=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE/(EPOCHS_NUM*0.5))

model.compile(loss="mean_squared_error", optimizer=optim, metrics=["accuracy"])

aug=ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest"
)

print("Training Network")
H=model.fit_generator(
    aug.flow(train_X, train_Y, batch_size=BATCH_SIZE),
    validation_data=(test_X, test_Y),
    steps_per_epoch=train_X.shape[0]//BATCH_SIZE,
    epochs=EPOCHS_NUM,
    class_weight=class_weight,
    verbose=1
)

# serialize model to JSON
model_json = model.to_json()
with open("model_DBSRCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1_DBSRCNN.h5")
print("Saved model to disk")

path_to_model = "./model_DBSRCNN.h5"
predictions = model.predict(test_X, batch_size=BATCH_SIZE)
model.save(path_to_model)
