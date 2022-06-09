import cv2
import matplotlib.pyplot as plt
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from numpy import asarray
import numpy as np
import pyscreenshot as ImageGrab


def prepro_img(test_img):
    # test_img = load_img('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/real_test_images/test_12.jpg')
    test_img = img_to_array(test_img)
    test_img_cropped = test_img[184:376, :]  # [184:376, :]
    # test_img_cropped = test_img
    init_shape = test_img_cropped.shape

    # test_img_resized = cv2.resize(test_img_cropped, dsize=(512, 256))
    # test_img_resized = asarray(test_img_resized)

    # generate image from source
    tst = np.expand_dims(test_img_cropped, 0)
    # Rescale from [0, 255] to [-1, 1]
    tst = (tst - 127.5) / 127.5
    gen_image = model.predict(tst)
    output_img = gen_image[0, :, :, :]
    # Rescaling from [-1, 1] to [0, 1]
    output_img = (output_img + 1) / 2.0
    # output_img255 = output_img * 255.0
    # output_normalized_img = cv2.resize(output_img, dsize=(init_shape[1], init_shape[0]))

    # cv2.imshow('Original Image', cv2.cvtColor(test_img.astype('uint8'), cv2.COLOR_RGB2BGR))
    # cv2.imshow('Cropped Image', cv2.cvtColor(test_img_cropped.astype('uint8'), cv2.COLOR_RGB2BGR))
    # cv2.imshow('Resized Image', cv2.cvtColor(test_img_resized.astype('uint8'), cv2.COLOR_RGB2BGR))
    # cv2.imshow('Preprocessed Image', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    # cv2.imshow('Preprocessed Normalized Image', cv2.cvtColor(output_normalized_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    # For video stream
    plt.imshow(output_img)
    # plt.imshow(test_img.astype('uint8'))
    plt.pause(0.001)
    plt.show()


# load model
model = load_model('prepro_models/dataset2_1/model_026100.h5')  # 008700  017400  026100

# # Image
# # image prepro
# img = load_img('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/real_test_images/test_12.jpg')
# # cv2.imshow('Preprocessed Normalized Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# prepro_img(img)


# Video
# Youtube video stream (https://www.youtube.com/watch?v=FbKLE7uar9Y&t=18s)
plt.ion()
while True:
    img = ImageGrab.grab(bbox=(0, 435, 640, 915))  # (left_x, top_y, right_x, bottom_y)
    # img = ImageGrab.grab(bbox=(235, 1080, 950, 1280))  # cropped for better relationship with training dataset.
    img = np.array(img)
    prepro_img(img)


