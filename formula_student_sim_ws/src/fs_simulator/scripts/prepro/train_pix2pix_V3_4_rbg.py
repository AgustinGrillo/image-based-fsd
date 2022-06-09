
"""
Based on:
https://github.com/whoIsTheGingerBreadMan/YoutubeVideos/blob/master/GANS/GANS.ipynb
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/more_advanced/DCGAN/train.py
"""


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


# Hyperparameters
RETRAINING_DEPTH = False
RETRAINING_MODEL_PATH = None
DATASET = 'data_set_5'
EPOCHS = 300
BATCH_SIZE = 8
VALIDATION_SPLIT = 1000
SHUFFLE_BUFFER = 15000  # Approx the size of the full dataset.
LAMBDA_GAN = 1
LAMBDA_SL_SIM = 1
LAMBDA_SL_DEPTH = 300
NOISE_PATH = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/'+DATASET+'/training_set/Noise'


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
    # output - RGB
    g_sim = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d5)
    out_image_sim = Activation('tanh')(g_sim)
    # output - Depth
    g_depth = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d5)
    out_image_depth = Activation('tanh')(g_depth)
    # define model
    model = Model(in_image, [out_image_sim, out_image_depth])
    return model


def generator_loss(target_sim, generated_sim, generated_output, target_depth, generated_depth):
    lambda_gan = LAMBDA_GAN
    lambda_sim = LAMBDA_SL_SIM
    lambda_depth = LAMBDA_SL_DEPTH

    # Gan loss
    gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(generated_output), logits=generated_output)
    gan_loss = tf.math.reduce_mean(gan_loss)

    # Supervised loss for simplified generated image
    supervised_loss_sim = tf.losses.mean_pairwise_squared_error(labels=target_sim, predictions=generated_sim)
    # supervised_loss = tf.losses.absolute_difference(labels=target, predictions=generated_output)  # L1?

    # Supervised loss for generated depth
    supervised_loss_depth = tf.losses.mean_squared_error(labels=target_depth, predictions=generated_depth)

    total_loss = lambda_gan*gan_loss + lambda_sim*supervised_loss_sim + lambda_depth*supervised_loss_depth

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


# Load the numpy files
def np_map_func(feature_path, flip):
    feature = np.load(feature_path)
    feature_crop = feature[184:376, :]
    if flip:
        feature_crop = np.fliplr(feature_crop)
    feature_crop = np.expand_dims(feature_crop, -1)
    return feature_crop


def np_noise_func(img_in):
    lim = np.random.uniform(0, 0.15)
    noise = np.random.uniform(-lim, lim, img_in.shape)
    noise = noise.astype(np.float32)
    noised_image = img_in + noise
    np.putmask(noised_image, noised_image < 0.0, 0.0)
    np.putmask(noised_image, noised_image > 1.0, 1.0)
    return noised_image


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
    # noise_image = tf.image.random_hue(noise_image, 0.03)
    # noise_image = tf.numpy_function(np_noise_func, [noise_image], tf.float32)
    # noise_image = tf.image.random_saturation(noise_image, 0.6, 1.0)
    # noise_image = tf.image.random_brightness(noise_image, 0.2)
    # Normalization: we scale it to [-1, 1]. (Note: that converting to float32 pre-scaled to [0, 1])
    noise_image = 2 * noise_image - 1.0

    # read the respective target image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]
    mid = tf.string_split([imagePath], os.path.sep, result_type='RaggedTensor').values
    file_name = mid[-1]
    img_num = tf.string_split([file_name], '.', result_type='RaggedTensor').values
    img_num = img_num[-2]
    mid = mid[-2]
    target_path = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/'+DATASET+'/training_set/Target_RGB/' + mid + '/' + img_num + '.jpg'
    target_image = tf.io.read_file(target_path)
    target_image = tf.image.decode_png(target_image, channels=3)
    target_image = tf.image.crop_to_bounding_box(target_image, offset_height=184, offset_width=0, target_height=192, target_width=640)
    target_image = tf.image.convert_image_dtype(target_image, dtype=tf.float32)
    target_image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(target_image), lambda: target_image)
    # TODO: Check Normalization. It must be coherent with tanh activation function used in U-Net.
    # Normalization: we scale it to [-1, 1]. (Note: that converting to float32 pre-scaled to [0, 1])
    target_image = 2 * target_image - 1.0

    # read the respective depth image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]
    depth_path = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/'+DATASET+'/training_set/Target_Depth/' + mid + '/' + img_num + '.npy'
    depth_npy = tf.numpy_function(np_map_func, [depth_path, flip_cond], tf.float32)
    depth_npy = depth_npy/50.0
    # TODO: Check Normalization. It must be coherent with tanh activation function used in U-Net.
    # Normalization: we scale it to [-1, 1]. (Note: that converting to float32 pre-scaled to [0, 1])
    depth_npy = 2 * depth_npy - 1.0

    return noise_image, target_image, depth_npy



# grab the list of images in our dataset directory and grab all
# unique class names
print("[INFO] loading image paths...")
NoiseImagePaths = list(paths.list_images(NOISE_PATH))

full_dataset = tf.data.Dataset.from_tensor_slices(NoiseImagePaths)
full_dataset = full_dataset.shuffle(SHUFFLE_BUFFER)
test_dataset_raw = full_dataset.take(VALIDATION_SPLIT)
train_dataset_raw = full_dataset.skip(VALIDATION_SPLIT)

train_dataset = (
    train_dataset_raw
    .shuffle(SHUFFLE_BUFFER)
    .map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # .cache()
    # .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)  # 2*BATCH_SIZE)  #tf.data.experimental.AUTOTUNE)
)
test_dataset = (
    test_dataset_raw
    .shuffle(VALIDATION_SPLIT)
    .map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # .cache()
    # .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)  # 2*BATCH_SIZE)  #tf.data.experimental.AUTOTUNE)
)

### DEBUG  -  PIPELINE ####
# def debug_show(image):
#     plt.figure()
#     plt.imshow(image)
#
# for img_n, img_t, img_d in train_dataset.take(1):
#     # debug_show(img_n[0])
#     # debug_show(img_t[0])
#     # debug_show(img_d[0, :, :, 0])
#     debug_show(0.5*(img_n[0]+1.0))
#     debug_show(0.5*(img_t[0]+1.0))
#     debug_show(0.5*(img_d[0, :, :, 0]+1.0))
### DEBUG  -  PIPELINE ####


discriminator = define_discriminator((192, 640, 3))
if RETRAINING_DEPTH:
    generator = tf.keras.models.load_model(RETRAINING_MODEL_PATH)
else:
    generator = define_generator((192, 640, 3))

print(discriminator.summary())
print(generator.summary())

generator_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)

loss_fn = keras.losses.BinaryCrossentropy()


def train_step(images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images[0], training=True)
        real_output = discriminator(images[1], training=True)
        generated_output = discriminator(generated_images[0], training=True)

        gen_loss = generator_loss(images[1], generated_images[0], generated_output, images[2], generated_images[1])
        disc_loss = discriminator_loss(real_output, generated_output)

    if RETRAINING_DEPTH:
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip([gradients_of_generator[42], gradients_of_generator[43]], [generator.trainable_variables[42], generator.trainable_variables[43]]))
    else:
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # print('G_loss: ', gen_loss.numpy(), 'D_loss: ', disc_loss.numpy())


def loss_calculator(images):
    generated_images = generator(images[0], training=False)
    real_output = discriminator(images[1], training=False)
    generated_output = discriminator(generated_images[0], training=False)
    gen_loss = generator_loss(images[1], generated_images[0], generated_output, images[2], generated_images[1])
    disc_loss = discriminator_loss(real_output, generated_output)
    return gen_loss, disc_loss


def train(train_dataset, test_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for idx, images in tqdm(enumerate(train_dataset)):
            train_step(images)

        # Calculate losses on test dataset
        gen_loss = 0
        disc_loss = 0
        for idx, test_images in enumerate(test_dataset):
            reduced_gen_loss, reduced_disc_loss = loss_calculator(test_images)
            gen_loss += reduced_gen_loss.numpy()
            disc_loss += reduced_disc_loss.numpy()
        gen_loss = gen_loss/(idx+1)
        disc_loss = disc_loss/(idx+1)

        if epoch % 10 == 0 or epoch == (epochs-1):
            fig, axs = plt.subplots(3, 3)
            plt.subplots_adjust(hspace=0)
            test_images = next(iter(test_dataset))
            for i_img in range(3):
                orig_img = tf.image.convert_image_dtype(0.5*(test_images[0][i_img]+1.0), dtype=tf.uint8)
                orig_img = orig_img.numpy()
                axs[0, i_img].imshow(orig_img)
                axs[0, i_img].axis('off')
                fake_sim = generator(test_images[0], training=False)[0][i_img]
                fake_sim = tf.image.convert_image_dtype(0.5*(fake_sim+1.0), dtype=tf.uint8)
                fake_sim = fake_sim.numpy()
                axs[1, i_img].imshow(fake_sim)
                axs[1, i_img].axis('off')
                fake_depth = generator(test_images[0], training=False)[1][i_img]
                fake_depth = tf.image.convert_image_dtype(0.5*(fake_depth+1.0), dtype=tf.uint8)
                fake_depth = fake_depth.numpy()
                axs[2, i_img].imshow(fake_depth[:, :, 0])
                axs[2, i_img].axis('off')
            fig.savefig('test_train_pix2pix/evaluation_epoch_{}'.format(epoch), bbox_inches='tight', dpi=400)


        # Saving the model
        if ((epoch % 10 == 0) & (epoch != 0)) or epoch == (epochs-1):
            generator.save('test_train_pix2pix/generator_model_epoch_{}'.format(epoch))
            # tst = tf.keras.models.load_model('test_train_pix2pix/generator_model_epoch_{}'.format(epoch))

        print ('Epoch: {}   Time Taken: {}  GenLoss: {} DiscLoss: {}'.format(epoch + 1, time.time()-start, gen_loss, disc_loss))


train(train_dataset, test_dataset, EPOCHS)
