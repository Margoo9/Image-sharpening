import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python import keras as K

from dataset.dataset_handling import load_data


EPOCHS_NUM = 50
INIT_LEARNING_RATE = 1e-4
BATCH_SIZE = 64


data = load_data('./dataset/Train')
train_Y, train_X = data['sharp'], data['blur']

data_test = load_data('./dataset/Test')
test_Y, test_X = data['sharp'], data['blur']


train_X = train_X.astype("float32")
test_X = test_X.astype("float32")
train_Y = train_Y.astype("float32")
test_Y = test_Y.astype("float32")


model = Sequential()
model.add(Input(shape=(256, 256, 1)))

model.add(Conv2D(32, (9, 9), activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(1, (5, 5), padding='same'))

model.compile(optimizer='adam', loss='mse', lr=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE/(EPOCHS_NUM*0.5),
              beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.build()
model.summary()

early_stopping = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_Y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS_NUM,
                    validation_data=(test_X, test_Y),
                    callbacks=[early_stopping]
                    )

# serialize model to JSON
model_json = model.to_json()
with open("model_DBSRCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_DBSRCNN_weights.h5")
model.save('DBSRCNN_model_blur.h5')
print("Saved model to disk")

path_to_model = "./model_DBSRCNN_predict.h5"
path_to_model_weigths = "./model_DBSRCNN_predict_weights.h5"
predictions = model.predict(test_X, batch_size=BATCH_SIZE)
model.save(path_to_model)
model.save_weights(path_to_model_weigths)
