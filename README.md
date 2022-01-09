# Image-sharpening

##### Project goals
The purpose of the project was to check whether the algorithms using deep neural networks have better efficiency in removing blur if the images were recorded using a specific optical system (and a specific image sensor). The project focused mainly on removing blur caused by the lens.

##### Technologies used
All code was written in Python 3.7. All of the proposed neural networks were written with Keras API and TensorFlow library. In order to maintain reasonable training time, the TensorFlow GPU was used.

##### Implementation

The project implementation was based on the following steps:
1. Prepare dataset.
2. Make a blurred photo of each sharp image in the dataset.
3. Write 4 different convolutional neural networks in order to check which one is the best.
4. Test the networks.
5. Check the networks' performance and save the resulting images.
6. Check the blur on the results.
7. Compare the blur on the photos made with a specific optical system and on the photos made with other devices.

##### Results achieved
Results showed the convolutional neural networks perform better when trained and tested on photos taken with a specific optical device.
