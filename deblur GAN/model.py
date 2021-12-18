import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, concatenate, Activation, Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Lambda
from keras.utils.vis_utils import plot_model
import numpy as np


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


# model

def generator_model():
    pass


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


if __name__ == '__main__':
    g = generator_model()
    g.summary()
    d = discriminator_model()
    d.summary()
    plot_model(d)
    m = generator_containing_discriminator(generator_model(), discriminator_model())
    m.summary()
