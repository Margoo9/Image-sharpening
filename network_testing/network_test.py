import os
import numpy as np
from PIL import Image
from dataset.dataset_handling import normalize_image, deprocess_image, load_image
import tensorflow.keras.models


# model - chosen network (GAN, DBSRCNN or our own)
def test_network(model, weight_path, input_dir, output_dir):
    model.load_weights(weight_path)
    for image_name in os.listdir(input_dir):
        image = np.array([normalize_image(load_image(os.path.join(input_dir, image_name)))])
        x_test = image
        generated_images = model.predict(x=x_test)
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(x_test)
        for i in range(generated_images.shape[0]):
            x = x_test[i, :, :, :]
            img = generated[i, :, :, :]
            output = np.concatenate((x, img), axis=1)
            im = Image.fromarray(output.astype(np.uint8))
            im.save(os.path.join(output_dir, image_name))


test_before_path = os.path.join('images_before_test')
test_before_path_with_image = os.path.join('images_before_test', 'testowa.jpg')
test_after_path = os.path.join('images_after_test')
test_weights_path = os.path.join('model1.h5')
test_model_path = os.path.join('./model.h5')
# prepare image (load and normalize)
test_image = load_image(test_before_path_with_image)
test_image = normalize_image(test_image)
# load model
test_model = tensorflow.keras.models.load_model(test_model_path)
# load weights, launch the network, deprocess and save image all in one
test_network(test_model, test_weights_path, test_before_path, test_after_path)
