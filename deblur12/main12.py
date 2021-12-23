import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python import keras as K

from dataset.dataset_handling import load_data_gray


EPOCHS_NUM = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 20


data = load_data_gray('../dataset/Train')
train_Y, train_X = data['sharp'], data['blur']

data_test = load_data_gray('../dataset/Test')
test_Y, test_X = data['sharp'], data['blur']


# scaling images to values [0 ... 1]
train_X = train_X.astype("float32") / 255.0
test_X = test_X.astype("float32") / 255.0

images_blur = np.reshape(train_X[1], (100, 100)) * 255.0
images_sharp = np.reshape(train_Y[1], (100, 100)) * 255.0


model = Sequential()
model.add(Input(shape=(100, 100, 1)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.compile(loss='mse', optimizer=K.optimizers.Adam(lr=LEARNING_RATE))
model.build()
model.summary()

early_stopping = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_Y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS_NUM,
                    validation_data=(test_X, test_Y),
                    callbacks=[early_stopping]
                    )

# history = model.fit(train_X, train_Y,
#                     batch_size=20,
#                     epochs=200,
#                     validation_data=(test_X, test_X),
#                     callbacks=[early_stopping]
#                     )

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_nn.h5")
print("Saved model to disk")

path_to_model = "./model_nn_predict.h5"
predictions = model.predict(test_X, batch_size=BATCH_SIZE)
model.save(path_to_model)

# predictions_0 = predictions[0] * 255.0
# predictions_0 = predictions_0.reshape(258, 540)
# plt.imshow(predictions_0, cmap='gray')


# EPOCHS_NUM = 30
# INIT_LEARNING_RATE = 1e-4
# BATCH_SIZE = 128
#
#
# model = Sequential()
# channel_dim = -1
#
# model.add(Conv2D(8, (5, 5), padding="same", activation='relu', input_shape=(32, 32, 3)))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(Conv2D(32, (3, 3), padding="same", strides=(1, 1), activation='relu'))
# model.add(BatchNormalization(axis=channel_dim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
#
# model.add(Dense(labels_num, activation='softmax'))
#
#
# optim=Adam(lr=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE/(EPOCHS_NUM*0.5))
#
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
#
# aug=ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.15,
#     horizontal_flip=False,
#     vertical_flip=False,
#     fill_mode="nearest"
# )
#
# print("Training Network")
# H=model.fit_generator(
#     aug.flow(train_X, train_Y, batch_size=BATCH_SIZE),
#     validation_data=(test_X, test_Y),
#     steps_per_epoch=train_X.shape[0]//BATCH_SIZE,
#     epochs=EPOCHS_NUM,
#     class_weight=class_weight,
#     verbose=1
# )
#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model1.h5")
# print("Saved model to disk")
#
# path_to_model = "./model.h5"
# predictions = model.predict(test_X, batch_size=BATCH_SIZE)
# model.save(path_to_model)
