import os
import dataset.dataset_handling as dh
import tensorflow.keras.models
import fajna_funkcja as fajna

test_before_path = os.path.join('../images_before_test')
test_before_path_with_image = os.path.join('../images_before_test', 'testowa.jpg')
test_after_path = os.path.join('../images_after_test')
test_weights_path = os.path.join('model1.h5')
test_model_path = os.path.join('./model.h5')
# prepare image (load and normalize)
test_image = dh.load_image(test_before_path_with_image)
test_image = dh.normalize_image(test_image)
# load model
test_model = tensorflow.keras.models.load_model(test_model_path)
# load weights, launch the network, deprocess and save image all in one
fajna.test_network(test_model, test_weights_path, test_before_path, test_after_path)
