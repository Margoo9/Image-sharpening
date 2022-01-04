import numpy as np
from PIL import Image
import os
import cv2
import glob as gb
from tensorflow.python import keras as K
import matplotlib.pyplot as plt

from dataset.dataset_handling import load_image, deprocess_image, normalize_image, load_images_100
from deblurGAN.model import generator_model


# is_size_100
# True if deblur neural network is used
# False if DBSRCNN or deblurGAN



# is_GAN
# True if GAN is tested
# False if other networks
is_GAN = False

# deblur
# is_size_100 = True
# path_to_test_image = '../network_testing/images_before_test/381.jpg'
# path_to_model = '../deblur12/model_nn.h5'
# path_to_model_weights = '../deblur12/model_nn_weights.h5'
# path_to_result_save = '../network_testing/images_after_test/'

# DBSRCNN
# is_size_100 = False
# path_to_test_image = '../network_testing/images_before_test'
# path_to_model = '../DBSRCNN/DBSRCNN_model_blur.h5'
# path_to_model_weights = '../DBSRCNN/model_DBSRCNN_weights.h5'
# path_to_result_save = '../network_testing/images_after_test'

# DBSRCNN
is_size_100 = False
path_to_test_image = '../network_testing/images_before_test'
path_to_model = '../DBSRCNN/model_nn.h5'
path_to_model_weights = '../DBSRCNN/model_nn_weights.h5'
path_to_result_save = '../network_testing/images_after_test'

batch_size = 4


def network_test(model, weights, test_image, out_dir, batch_size):
    if is_size_100:
        test_img = gb.glob(test_image)
        x_test = load_images_100(test_img)

        chosen_model = K.models.load_model(model)
        chosen_model.load_weights(weights)

        res = chosen_model.predict(x_test)
        res_0 = res[0] * 255.0
        res_0 = res_0.reshape(100, 100)
        # plt.imshow(res_0, cmap='gray')
        # plt.imshow(preds_0)
        # plt.show()
        # cv2.imwrite('result.png', res_0)
        im = Image.fromarray(res_0.astype(np.uint8))
        im.save(os.path.join(out_dir, 'result.jpg'))
    elif is_GAN:
        for image_name in os.listdir(test_image):
            image = np.array([normalize_image(load_image(os.path.join(test_image, image_name)))])
            x_test = image
            chosen_model = generator_model()
            chosen_model.load_weights(weights)
            generated_images = chosen_model.predict(x=x_test)
            generated = np.array([deprocess_image(img) for img in generated_images])
            x_test = deprocess_image(x_test)
            for i in range(generated_images.shape[0]):
                x = x_test[i, :, :, :]
                img = generated[i, :, :, :]
                output = np.concatenate((x, img), axis=1)
                im = Image.fromarray(output.astype(np.uint8))
                im.save(os.path.join(out_dir, image_name))
    else:
        for image_name in os.listdir(test_image):
            image = np.array([normalize_image(load_image(os.path.join(test_image, image_name)))])
            x_test = image
            chosen_model = K.models.load_model(model)
            chosen_model.load_weights(weights)
            generated_images = chosen_model.predict(x=x_test)
            generated = np.array([deprocess_image(img) for img in generated_images])
            x_test = deprocess_image(x_test)
            for i in range(generated_images.shape[0]):
                # x = x_test[i, :, :, :]
                img = generated[i, :, :, :]
                # output = np.concatenate((x, img), axis=1)
                # im = Image.fromarray(output.astype(np.uint8))
                im = Image.fromarray(img.astype(np.uint8))
                im = np.array(im)
                # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                im.save(os.path.join(out_dir, 'result.jpg'))


network_test(path_to_model, path_to_model_weights, path_to_test_image, path_to_result_save, batch_size)

