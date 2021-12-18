import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, concatenate, Activation, Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Lambda
import numpy as np

from dataset.dataset_handling import load_data


train_X, train_Y = load_data("dataset/Train")
test_X, test_Y = load_data("dataset/Test")

image_shape = (256, 256, 3)
channel_rate = 64
patch_shape = (channel_rate, channel_rate, 3)


# losses
def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


# models

def discriminator_model():
    sigmoid_as_activation = False
    inputs = Input(shape=patch_shape)
    x = Conv2D(filters=channel_rate, kernel_size=(4, 4), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=2 * channel_rate, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=4 * channel_rate, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=4 * channel_rate, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    if sigmoid_as_activation:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model


def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


generator = Sequential()
channel_dim = -1
generator.add(Dense(7 * 7 * 128, input_shape=(32, 32, 3)))
generator.add(Reshape([7, 7, 128]))
generator.add(BatchNormalization(axis=channel_dim))
generator.add(Conv2DTranspose(64, (5, 5), strides=2, padding="same", activation='selu'))
generator.add(BatchNormalization(axis=channel_dim))
generator.add(Conv2DTranspose(1, (5, 5), strides=2, padding="same", activation='tanh'))

discriminator = Sequential()
channel_dim = -1
discriminator.add(Conv2D(64, (5, 5), strides=2, padding="same", activation=LeakyReLU(0.2), input_shape=(28, 28, 1)))
discriminator.Dropout(0.4)
discriminator.add(Conv2D(128, (5, 5), strides=2, padding="same", activation=LeakyReLU(0.2)))
discriminator.Dropout(0.4)
discriminator.Flatten()
discriminator.add(Dense(1, activation='sigmoid'))

gan = Sequential([generator, discriminator])

discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

EPOCHS_NUM = 30
INIT_LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# dataset = tf.data.Dataset.form_tensor_slices(train_X).shuffle(1000)
# d1ataset = dataset.batch(BATCH_SIZE, drop_reminder=True).prefetch(1)


def train_network(gan, coding_size):
    generator, discriminator = gan.layers
    for epoch in range(EPOCHS_NUM):
        for batch_X in train_X:
            # Phase 1 - discriminator's learning
            noise = tf.random.normal(shape=[BATCH_SIZE, coding_size])
            generated_images = generator(noise)
            fake_and_not_images = tf.concat([generated_images, batch_X], axis=0)
            y1 = tf.constant([[0.]] * BATCH_SIZE + [[1.]] * BATCH_SIZE)
            discriminator.trainable = True
            discriminator.train_on_batch(fake_and_not_images, y1)
            # Phase 2 - generator's learning
            noise = tf.random.normal(shape=[BATCH_SIZE, coding_size])
            y2 = tf.constant([[1.]] * BATCH_SIZE)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)


train_network(gan, coding_size=30)
