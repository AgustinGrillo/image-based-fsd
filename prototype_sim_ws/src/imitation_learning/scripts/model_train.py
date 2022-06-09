#!/usr/bin/env python


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import cv2
import keras.backend
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers
from keras import Input, layers, models, callbacks, constraints
# from keras import backend as K
# from keras.models import Model, Sequential
# from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Lambda, Cropping2D
# from keras.layers.convolutional import Convolution2D
# from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split
import image_preprocessing
import rospkg


rospack = rospkg.RosPack()
path_imitation_learning = rospack.get_path('imitation_learning')

### Variables to modify ###
RE_TRAINING = False
LOAD_WEIGHTS = False

batch_size_value = 128  # 32 64 1024
n_epoch = 200  # 10
initial_epoch = 0

version = "10"
subversion = "1"

filepath_model_load = path_imitation_learning + "/models/v6/driverless_model.h5"
filepath_model_weights_load = path_imitation_learning + "/weights/v2/weights-improvement-v2.5-29-0.72.hdf5"

data_training = ["/output/data_101", "/output/data_103", "/output/data_104", "/output/data_106", "/output/data_107", "/output/data_108",
                 "/output/data_109", "/output/data_110", "/output/data_113", "/output/data_114", "/output/data_115", "/output/data_116",
                 "/output/data_117"]
data_validation = ["/output/data_102", "/output/data_105", "/output/data_111", "/output/data_112", "/output/data_118"]

skip_line_train = 2  # 4
skip_line_valid = 2  # 10  # Cuantos frames me salteo

screen_size = [360, 640, 3]

###########################
# GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
# tf.compat.v1.keras.backend.set_session(sess)
keras.backend.set_session(sess)

###########################

if not os.path.exists(path_imitation_learning + "/models/v" + version):
    os.mkdir(path_imitation_learning + "/models/v" + version)
if not os.path.exists(path_imitation_learning + "/weights/v" + version):
    os.mkdir(path_imitation_learning + "/weights/v" + version)

filepath_model_save = path_imitation_learning + "/models/v" + version + "/driverless_model_v" + version + "." + subversion + ".h5"
filepath_tensorboard = path_imitation_learning + "/logs/run" + version + "." + subversion
filepath_weights_checkpoint = path_imitation_learning + "/weights/v" + version + "/weights-improvement-v" + version + "." + subversion + "-{epoch:02d}-{val_loss:.2f}.hdf5"

print("Model save path:", filepath_model_save)

### GRAB TRAINING DATA ####

train_samples = []
for data_name in data_training:

    filepath_data = path_imitation_learning + data_name + "/interpolated.csv"
    filepath_images = path_imitation_learning + data_name + "/center/"

    print("Training Paths:", filepath_data)

    with open(filepath_data) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[5] != "filename":
                path = os.path.normpath(line[5]).split(os.path.sep)
                line[5] = filepath_images + path[1].split(
                    '\\')[-1]
                train_samples.append(line)

train_samples = train_samples[0:len(train_samples):skip_line_train]
train_samples = sklearn.utils.shuffle(train_samples)

### GRAB VALIDATION DATA ####

validation_samples = []
for data_name in data_validation:

    filepath_data = path_imitation_learning + data_name + "/interpolated.csv"  # '/home/agus/ros/fss_ws/src/imitation_learning/output/data_1/interpolated.csv'
    filepath_images = path_imitation_learning + data_name + "/center/"  # '/home/agus/ros/fss_ws/src/imitation_learning/output/data_1/center/'

    print("Validation Paths:", filepath_data)

    with open(filepath_data) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[5] != "filename":
                path = os.path.normpath(line[5]).split(os.path.sep)
                line[5] = filepath_images + path[1].split(
                    '\\')[-1]
                validation_samples.append(line)

validation_samples = validation_samples[0:len(validation_samples):skip_line_valid]
validation_samples = sklearn.utils.shuffle(validation_samples)

print("Number of training samples: ", len(train_samples))
print("Number of validation samples: ", len(validation_samples))


# index,timestamp,width,height,frame_id,filename,angle,speed
def generator(samples, batch_size=32, aug=0):  # aug: augmentation
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            # print(batch_samples)
            images = []
            outputs = []
            for batch_sample in batch_samples:
                if batch_sample[5] != "filename":
                    # path = os.path.normpath(batch_sample[5]).split(os.path.sep)
                    # name = filepath_images + path[1].split(
                    #    '\\')[-1]
                    center_image = cv2.imread(batch_sample[5])

                    # TODO: Scale accordingly
                    # Normalized output
                    # Angle: [-1, 1] --> [-1, 1]
                    angle = float(batch_sample[6])
                    # Speed: [-10, 10] --> [-1, 1]
                    speed = float(batch_sample[7]) / 0.3

                    images.append(center_image)
                    outputs.append([angle, speed])

                    if aug:
                        flip_image = np.fliplr(center_image)
                        flip_angle = -1 * angle
                        images.append(flip_image)
                        outputs.append([flip_angle, speed])

            x_train = np.array(images)
            y_train = np.array(outputs)
            if len(x_train.shape) < 4:
                x_train = np.expand_dims(x_train, axis=3)

            yield sklearn.utils.shuffle(x_train, y_train)


# compile and train the model using the generator function

train_generator = generator(train_samples, batch_size=batch_size_value, aug=0)
validation_generator = generator(
    validation_samples, batch_size=batch_size_value, aug=0)

# Load model for re-training or create a new one
if RE_TRAINING is True:
    model = models.load_model(filepath_model_load)
    if LOAD_WEIGHTS is True:
        model.load_weights(filepath_model_weights_load)

else:

    inputs = Input(shape=screen_size)
    x = layers.Lambda(lambda m: (m / 255.0) - 0.5)(inputs)
    # x = layers.Cropping2D(cropping=((184, 133), (0, 0)), input_shape=screen_size)(x)
    x = layers.Conv2D(filters=4, kernel_size=(5, 5), activation="relu", name="conv_1", strides=(2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=8, kernel_size=(5, 5), activation="relu", name="conv_2", strides=(2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", name="conv_3", strides=(1, 1))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.SpatialDropout2D(.5, dim_ordering='default')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", name="conv_4", strides=(1, 1))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv_5", strides=(1, 1))(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu", name="dense_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_2")(x)
    x = layers.Dense(64, activation="relu", name="dense_3")(x)
    x = layers.Dense(10, activation="relu", name="dense_4")(x)
    # Output layer
    outputs = layers.Dense(2, activation='linear', name="dense_5")(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    # Model compile
    model.compile(
        loss='mse',
        optimizer=optimizers.adam(lr=0.0002),  # 'adam' or 'SGD' or 'Nadam'
        metrics=['mse']
    )

# Model visualization
model.summary()
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# Chekpoints
callbacks = [
    callbacks.ModelCheckpoint(filepath=filepath_weights_checkpoint, save_best_only=True, period=2),
    callbacks.TensorBoard(log_dir=filepath_tensorboard)   # to run tensorboard, execute in a terminal (in respective folder): tensorboard --logdir=./logs or tensorboard --logdir logs and open link in browser.
]

# PROBAR HE-NORMAL INITIALIZATION
history = model.fit_generator(
    train_generator, steps_per_epoch=(len(train_samples) / batch_size_value), epochs=n_epoch, verbose=2,
    validation_data=validation_generator, validation_steps=(len(validation_samples) / batch_size_value),
    callbacks=callbacks,
    initial_epoch=initial_epoch,
    class_weight={0: 7, 1: 3}
)

# class_weight: le bajamos la importancia al  comando 'speed', dado que es el que mas se utiliza.
# Esto es para contrarrestar el data imbalance.
# Recordar: clases -> ('angle', 'speed')

# Save model
model.save(filepath_model_save)

# with open('models/v2/driverless_model.json', 'w') as output_json:
#     output_json.write(model.to_json())
#

# # Save TensorFlow model
# tf.train.write_graph(
#     K.get_session().graph.as_graph_def(),
#     logdir='.',  # logdir='.'
#     name=path_imitation_learning + "/models/v" + version + "/driverless_model_v" + version + "." + subversion + ".pb",
#     as_text=False)

print('Done training')


#### PRINT OUT SUMMARY OF TRAINING ####

# path_split = filepath_model_save.split('/')
#
# path_txt = ""
#
# for i in range(1, len(path_split)):
#     path_txt = path_txt + '/' + path_split[i]
#     if path_split[i - 1] == "models":
#         model_name = path_split[i + 1]
#         break
#
# print path_txt
#
# f = open(path_txt + "/README.txt", "a+")
#
# f.write("-------------------------------------------------------- \n\n")
#
# f.write("Model Name: " + model_name + "\n\n")
#
# f.write("Trained in: \n")
#
# for data_name in data_training:
#     f.write("   -" + data_name + "\n")
#
# f.write("Validated  in: \n")
#
# for data_name in data_validation:
#     f.write("   -" + data_name + "\n")
#
# f.write("\n Epochs:  %d   \n" % n_epoch)
# f.write("\n Batch Sample Size: %d \n " % batch_size_value)
#
# f.write("\n\n\n")
#
# f.close()
