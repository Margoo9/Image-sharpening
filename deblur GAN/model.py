import tensorflow.python.keras.backend as K
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, concatenate, Activation, Conv2D, BatchNormalization, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.utils.vis_utils import plot_model


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

def dense_block(inputs, dilation_factor=None):
    x = LeakyReLU(alpha=0.2)(inputs)
    x = Conv2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    if dilation_factor is not None:
        x = Conv2D(filters=channel_rate, kernel_size=(3, 3), padding='same', dilation_rate=dilation_factor)(x)
    else:
        x = Conv2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    return x


def generator_model():
    inputs = Input(shape=(None, None, 3))
    h = Conv2D(filters=4 * channel_rate, kernel_size=(3, 3), padding='same')(inputs)

    dense_1 = dense_block(inputs=h)
    x = concatenate([h, dense_1])
    dense_2 = dense_block(inputs=x, dilation_factor=(1, 1))
    x = concatenate([x, dense_2])
    dense_3 = dense_block(inputs=x)
    x = concatenate([x, dense_3])
    dense_4 = dense_block(inputs=x, dilation_factor=(2, 2))
    x = concatenate([x, dense_4])
    dense_5 = dense_block(inputs=x)
    x = concatenate([x, dense_5])
    dense_6 = dense_block(inputs=x, dilation_factor=(3, 3))
    x = concatenate([x, dense_6])
    dense_7 = dense_block(inputs=x)
    x = concatenate([x, dense_7])
    dense_8 = dense_block(inputs=x, dilation_factor=(2, 2))
    x = concatenate([x, dense_8])
    dense_9 = dense_block(inputs=x)
    x = concatenate([x, dense_9])
    dense_10 = dense_block(inputs=x, dilation_factor=(1, 1))

    x = LeakyReLU(alpha=0.2)(dense_10)
    x = Conv2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x = concatenate([h, x])
    x = Conv2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    outputs = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


def discriminator_model():
    sigmoid_as_activation = False
    inputs = Input(shape=patch_shape)
    x = Conv2D(filters=channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=2 * channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=4 * channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=4 * channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
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
#     model = Model(inputs=inputs, outputs=outputs)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model


if __name__ == '__main__':
    g = generator_model()
    g.summary()
    d = discriminator_model()
    d.summary()
    plot_model(d)
    m = generator_containing_discriminator(generator_model(), discriminator_model())
    m.summary()
