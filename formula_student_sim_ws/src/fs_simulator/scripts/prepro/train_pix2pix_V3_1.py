
"""
Based on:
https://github.com/whoIsTheGingerBreadMan/YoutubeVideos/blob/master/GANS/GANS.ipynb
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/more_advanced/DCGAN/train.py
"""
# TODO: Add Discriminator Patches
# TODO: Manage dataset.

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

from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)


# load and prepare training images
def load_np_samples(filename):
    # load compressed arrays
    data = np.load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


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


dataset_np = load_np_samples('dataset_images/data_set_2/dataset_compressed.npz')
SHUFFLE_BUFFER_SIZE = dataset_np[0].shape[0]
BATCH_SIZE = 8
train_dataset = tf.data.Dataset.from_tensor_slices((dataset_np[0], dataset_np[1]))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

discriminator = define_discriminator(dataset_np[0].shape[1:])
generator = define_generator(dataset_np[0].shape[1:])

print(discriminator.summary())
print(generator.summary())

generator_optimizer = keras.optimizers.Adam(0.0002)
discriminator_optimizer = keras.optimizers.Adam(0.0002)

loss_fn = keras.losses.BinaryCrossentropy()

# Training
EPOCHS = 50


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
            train_step(images)

        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


train(train_dataset, EPOCHS)
