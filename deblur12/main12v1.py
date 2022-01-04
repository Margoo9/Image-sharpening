import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D, GlobalMaxPooling1D
# from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Input

from tensorflow.python import keras as K

# from dataset.dataset_handling import load_data_gray
from dataset.dataset_handling import load_data


EPOCHS_NUM = 75
LEARNING_RATE = 1e-4
BATCH_SIZE = 20

path_to_train_data = '../dataset/train'
path_to_test_data = '../dataset/test'

data = load_data(path_to_train_data)
train_Y, train_X = data['sharp'], data['blur']

data_test = load_data(path_to_test_data)
test_Y, test_X = data['sharp'], data['blur']


train_X = train_X.astype("float32")
test_X = test_X.astype("float32")
train_Y = train_Y.astype("float32")
test_Y = test_Y.astype("float32")
# path_to_train_data ='../dataset/train'
# path_to_test_data = '../dataset/test'
#
#
# # data = load_data_gray(path_to_train_data)
# data = load_data(path_to_train_data)
#
# train_Y, train_X = data['sharp'], data['blur']
#
# # data_test = load_data_gray(path_to_test_data)
# data_test = load_data(path_to_test_data)
# test_Y, test_X = data['sharp'], data['blur']


# scaling images to values [0 ... 1]
# train_X = train_X.astype("float32") / 255.0
# test_X = test_X.astype("float32") / 255.0
# train_Y = train_Y.astype("float32") / 255.0
# test_Y = test_Y.astype("float32") / 255.0

train_X = np.expand_dims(train_X, axis=-1)
test_X = np.expand_dims(test_X, axis=-1)
train_Y = np.expand_dims(train_Y, axis=-1)
test_Y = np.expand_dims(test_Y, axis=-1)

model = Sequential()
# model.add(Input(shape=(100, 100, 1)))
# kolorowe
model.add(Input(shape=(256, 256, 3)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), padding='same'))
model.add(GlobalMaxPooling1D())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Flatten())
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

model.compile(loss='mse', optimizer='adam', learning_rate=LEARNING_RATE)
model.build()
model.summary()

early_stopping = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_Y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS_NUM,
                    validation_data=(test_X, test_Y)
                    # callbacks=[early_stopping]
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
model.save_weights("model_nn_weights.h5")
print("Saved model to disk")

path_to_model = "./model_nn.h5"
path_to_model_weights = "./model_weights_pred.h5"
predictions = model.predict(test_X, batch_size=BATCH_SIZE)
model.save(path_to_model)
model.save_weights(path_to_model_weights)

# predictions_0 = predictions[0] * 255.0
# predictions_0 = predictions_0.reshape(258, 540)
# plt.imshow(predictions_0, cmap='gray')
