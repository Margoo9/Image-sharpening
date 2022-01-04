import cv2
import numpy as np
import os
from tensorflow.python import keras as K


def load_image(path_to_image):
    image = cv2.imread(path_to_image)
    return image


def load_image_gray(path_to_image):
    image = cv2.imread(path_to_image, 0)
    image = cv2.resize(image, (100, 100))
    return image


def load_images_100(path):
    image_list = np.zeros((len(path), 100, 100, 1))
    for i, fig in enumerate(path):
        # img = K.preprocessing.image.load_img(fig, target_size=(100, 100))
        img = K.preprocessing.image.load_img(fig, color_mode='grayscale', target_size=(100, 100))
        x = K.preprocessing.image.img_to_array(img).astype('float32')
        x = x / 255.0
        image_list[i] = x

    return image_list


# normalize data in range [-1, 1]
def normalize_image(image):
    # image = cv2.reshape(image, (256, 256), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (256, 256))
    image = np.array(image)
    image = (image - 127.5) / 127.5
    return image


def normalize_with_numpy(image):
    normalized_image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    normalized_image = 2 * normalized_image - 1
    return normalized_image


def deprocess_image(image):
    image = image * 127.5 + 127.5
    image = image.astype('uint8')
    return image


def save_image(image, path):
    image = image * 127.5 + 127.5
    cv2.imwrite(os.path.join(path, 'result.jpg'), image)
    
    
def get_images(path):
    images_list = []
    for file in os.listdir(path):
        if file.endswith(('.jpg', '.png', 'jpeg')):
            images_list.append(os.path.join(path, file))
    return images_list


def load_data(path):
    images_blur = []
    images_sharp = []
    blurs_path = os.path.join(path, 'blur')
    sharps_path = os.path.join(path, 'sharp')
    blurs_path, sharps_path = get_images(blurs_path), get_images(sharps_path)
    for x, y in zip(blurs_path, sharps_path):
        img_blur, img_sharp = load_image(x), load_image(y)
        images_blur.append(normalize_image(img_blur))
        images_sharp.append(normalize_image(img_sharp))
    return {
        'blur': np.array(images_blur), 
        'sharp': np.array(images_sharp)
    }

def load_data_gray(path):
    images_blur = []
    images_sharp = []
    blurs_path = os.path.join(path, 'blur')
    sharps_path = os.path.join(path, 'sharp')
    blurs_path, sharps_path = get_images(blurs_path), get_images(sharps_path)
    for x, y in zip(blurs_path, sharps_path):
        img_blur, img_sharp = load_image_gray(x), load_image_gray(y)
        images_blur.append(img_blur)
        images_sharp.append(img_sharp)
    return {
        'blur': np.array(images_blur), 
        'sharp': np.array(images_sharp)
    }

