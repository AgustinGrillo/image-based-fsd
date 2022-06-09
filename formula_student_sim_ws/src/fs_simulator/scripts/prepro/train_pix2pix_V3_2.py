
"""
Based on:
https://github.com/whoIsTheGingerBreadMan/YoutubeVideos/blob/master/GANS/GANS.ipynb
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/more_advanced/DCGAN/train.py
"""
# TODO: Add Discriminator Patches
# TODO: Add paper u-net and losses

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.backend import set_session
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

from imutils import paths
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)



# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # Image input
    in_image = Input(shape=image_shape)

    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model(in_image, patch_out)
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    # e6 = define_encoder_block(e5, 512)
    # e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e5, 512)
    d2 = decoder_block(d1, e4, 512)
    d3 = decoder_block(d2, e3, 256)
    d4 = decoder_block(d3, e2, 128, dropout=False)
    d5 = decoder_block(d4, e1, 64, dropout=False)
    # d6 = decoder_block(d5, e1, 64, dropout=False)
    # d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d5)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


def generator_loss(target, generated_image, generated_output):
    lambda_coef = 100
    gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(generated_output), logits=generated_output)
    gan_loss = tf.math.reduce_mean(gan_loss)
    supervised_loss = tf.losses.mean_pairwise_squared_error(labels=target, predictions=generated_image)
    # supervised_loss = tf.losses.absolute_difference(labels=target, predictions=generated_output)  # L1?
    total_loss = gan_loss + lambda_coef*supervised_loss

    return total_loss


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)
    real_loss = tf.math.reduce_mean(real_loss)
    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(generated_output), logits=generated_output)
    generated_loss = tf.math.reduce_mean(generated_loss)

    total_loss = real_loss + generated_loss

    return total_loss


def load_images(imagePath):
    uniform_random = tf.random_uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, .5)

    # read the Noise image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]
    noise_image = tf.io.read_file(imagePath)
    noise_image = tf.image.decode_png(noise_image, channels=3)
    noise_image = tf.image.crop_to_bounding_box(noise_image, offset_height=184, offset_width=0, target_height=192, target_width=640)
    noise_image = tf.image.convert_image_dtype(noise_image, dtype=tf.float32)
    # Augmentation
    noise_image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(noise_image), lambda: noise_image)
    noise_image = tf.image.random_hue(noise_image, 0.03)
    noise_image = tf.image.random_saturation(noise_image, 0.96, 1.0)
    noise_image = tf.image.random_brightness(noise_image, 0.08)
    # Normalization: we scale it to [-1, 1]. (Note: that converting to float32 pre-scaled to [0, 1])
    # noise_image = 2 * noise_image - 1.0

    # read the respective target image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]
    mid = tf.string_split([imagePath], os.path.sep, result_type='RaggedTensor').values
    mid = mid[-2]
    target_path = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/data_set_2/training_set/Target_Simplified/' + mid + '/' + mid + '_0.jpg'
    target_image = tf.io.read_file(target_path)
    target_image = tf.image.decode_png(target_image, channels=3)
    target_image = tf.image.crop_to_bounding_box(target_image, offset_height=184, offset_width=0, target_height=192, target_width=640)
    target_image = tf.image.convert_image_dtype(target_image, dtype=tf.float32)
    target_image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(target_image), lambda: target_image)
    # Normalization: we scale it to [-1, 1]. (Note: that converting to float32 pre-scaled to [0, 1])
    # target_image = 2 * target_image - 1.0

    return noise_image, target_image


# initialize batch size and number of steps
BATCH_SIZE = 8
NUM_STEPS = 1000
NOISE_PATH = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/data_set_2/training_set/Noise'

# grab the list of images in our dataset directory and grab all
# unique class names
print("[INFO] loading image paths...")
NoiseImagePaths = list(paths.list_images(NOISE_PATH))

dataset = tf.data.Dataset.from_tensor_slices(NoiseImagePaths)
train_dataset = (
    dataset
    .shuffle(1024)
    .map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # .cache()
    # .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)  # 2*BATCH_SIZE)  #tf.data.experimental.AUTOTUNE)
)

#### DEBUG  -  PIPELINE ####
# def debug_show(image):
#     plt.figure()
#     plt.imshow(image)
#
# for img_n, img_t in train_dataset.take(1):
#     debug_show(img_n[0])
#     debug_show(img_t[0])
#     # debug_show(0.5*(img_n[0]+1.0))
#     # debug_show(0.5*(img_t[0]+1.0))
#### DEBUG  -  PIPELINE ####


discriminator = define_discriminator((192, 640, 3))
generator = define_generator((192, 640, 3))

print(discriminator.summary())
print(generator.summary())

generator_optimizer = keras.optimizers.Adam(0.0002)
discriminator_optimizer = keras.optimizers.Adam(0.0002)

loss_fn = keras.losses.BinaryCrossentropy()

# Training
EPOCHS = 300


def train_step(images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images[0], training=True)
        real_output = discriminator(images[1], training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(images[1], generated_images, generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # print('G_loss: ', gen_loss.numpy(), 'D_loss: ', disc_loss.numpy())


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_loss = 500
        disc_loss = 500

        for idx, images in tqdm(enumerate(dataset)):
            if idx % 100 == 0:
                fake_img = generator(images[0], training=False)[0]
                fake_img = tf.image.convert_image_dtype(fake_img, dtype=tf.uint8)
                fake_img_png = tf.image.encode_png(fake_img)
                tf.write_file('test_train_pix2pix/test_img_{}_{}.png'.format(epoch, idx), fake_img_png)
            train_step(images)

        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


train(train_dataset, EPOCHS)
