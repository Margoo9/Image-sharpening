import cv2
import numpy as np
import os


def load_image(path_to_image):
    image = cv2.imread(path_to_image)
    return image


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
        if file.endswith(".jpg"):
            images_list.append(os.path.join(path, file))
    return images_list




