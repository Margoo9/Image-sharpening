import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np

from dataset.dataset_handling import load_data, deprocess_image
from model import generator_model, discriminator_model, generator_containing_discriminator, perceptual_loss, wasserstein_loss


lambda_val = 100

path_to_results = './results'


def train(batch_size, epoch_num, discriminator_train_num=5):
    data = load_data('./dataset/Train')
    y_train, x_train = data['sharp'], data['blur']

    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator(g, d)

    d.trainable = True
    d.compile(optimizer='adam', loss=wasserstein_loss, lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [lambda_val, 1]
    d_on_g.compile(optimizer='adam', lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                   loss=loss, loss_weights=loss_weights)
    d.trainable = True

    for epoch in tqdm(range(epoch_num)):
        # print("Training epoch {}".format(epoch + 1), '/', epoch_num)

        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index * batch_size:(index + 1) * batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(discriminator_train_num):
                d_loss_real = d.train_on_batch(image_full_batch, np.ones((batch_size, 1)))
                d_loss_fake = d.train_on_batch(generated_images, -np.ones((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                print('Batch %d d_loss : %f' % (index + 1, d_loss))
                d_losses.append(d_loss)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, np.ones((batch_size, 1))])
            print('Batch %d d_on_g_loss : %f' % (index + 1, d_on_g_loss))
            d_on_g_losses.append(d_on_g_loss)

            # g_loss = g.train_on_batch(image_blur_batch, image_full_batch)
            # print('batch %d g_loss : %f' % (index + 1, g_loss))

            d.trainable = True

        g.save_weights(os.path.join(WEIGHTS_DIR, 'generator_{}_{}.h5'.format(epoch, int(np.mean(d_on_g_losses)))), True)
        d.save_weights(os.path.join(WEIGHTS_DIR, 'discriminator_{}.h5'.format(epoch)), True)

        
def test(batch_size):
    data = load_data('./dataset/Test', batch_size)
    y_test, x_test = data['sharp'], data['blur']
    g = generator_model()
    g.load_weights('weights/generator.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('res{}.jpg'.format(i)) 
        
        
if __name__ == '__main__':
    train(16, 50)    
    test(4)
