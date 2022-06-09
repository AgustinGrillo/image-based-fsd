#!/usr/bin/env python

# Este codigo entrena una secuencia de imagenes.
# idealmente una secuencia de 6 (salteandose cada 3 a 10 FPS). Suponiendo una velocidad de crucero de 15 m/s del formula.
# Se deben poder ver conos anteriores.

# La red se entrena con 6 imagenes, siendo la imagen 1 la mas vieja, y la 6 la mas nueva.


import os
import csv
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import keras
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

batch_size_value = 16  # 32 64 1024
n_epoch = 1000  # 10
initial_epoch = 0

version = "S1"
subversion = "0"

# Number of frames to feed the sequence, and how many frames to skip.
# (if num_skip_frames=1, then the frames are consecutive)
num_frames = 5
num_skip_frames = 50
######################################

filepath_model_load = path_imitation_learning + "/models/sequence/vS1/driverless_model.h5"
filepath_model_weights_load = path_imitation_learning + "/weights/sequence/vS1/weights-improvement-v2.5-29-0.72.hdf5"

data_training = ["/output/data_1", "/output/data_2", "/output/data_3", "/output/data_5", "/output/data_9", "/output/data_10",
                 "/output/data_14", "/output/data_15_complicado", "/output/data_17"]
data_validation = ["/output/data_8", "/output/data_11", "/output/data_16_complicado", "/output/data_19"]

skip_line_train = 1  # 4
skip_line_valid = 1  # 10  # Cuantos frames me salteo

screen_size = [360, 640, 3]

###########################
# GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
# tf.compat.v1.keras.backend.set_session(sess)1
keras.backend.set_session(sess)

###########################

if not os.path.exists(path_imitation_learning + "/models/sequence/v" + version):
    os.makedirs(path_imitation_learning + "/models/sequence/v" + version)
if not os.path.exists(path_imitation_learning + "/weights/sequence/v" + version):
    os.makedirs(path_imitation_learning + "/weights/sequence/v" + version)

filepath_model_save = path_imitation_learning + "/models/sequence/v" + version + "/driverless_model_v" + version + "." + subversion + ".h5"
filepath_tensorboard = path_imitation_learning + "/logs/run" + version + "." + subversion
filepath_weights_checkpoint = path_imitation_learning + "/weights/sequence/v" + version + "/weights-improvement-v" + version + "." + subversion + "-{epoch:02d}-{val_loss:.2f}.hdf5"

print "Model save path:", filepath_model_save

### GRAB TRAINING DATA ####

train_samples = []
for data_name in data_training:

    filepath_data = path_imitation_learning + data_name + "/interpolated.csv"
    filepath_images = path_imitation_learning + data_name + "/center/"

    print "Training Paths:", filepath_data

    with open(filepath_data) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[5] != "filename":
                path = os.path.normpath(line[5]).split(os.path.sep)
                line[5] = filepath_images + path[1].split(
                    '\\')[-1]
                train_samples.append(line)

train_samples_sequence = []
for idx, element in enumerate(train_samples):
    # element = element[5:8]  # we grab only camera, speed and angle.
    if (idx <= ( len(train_samples) - ((num_frames-1)*num_skip_frames+1) )):
        (train_samples_sequence.append(train_samples[idx:idx+((num_frames-1)*num_skip_frames+1):num_skip_frames]))

train_samples_sequence = train_samples_sequence[0:len(train_samples_sequence):skip_line_train]
train_samples_sequence = sklearn.utils.shuffle(train_samples_sequence)

### GRAB VALIDATION DATA ####

validation_samples = []
for data_name in data_validation:

    filepath_data = path_imitation_learning + data_name + "/interpolated.csv"
    filepath_images = path_imitation_learning + data_name + "/center/"

    print "Validation Paths:", filepath_data

    with open(filepath_data) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[5] != "filename":
                path = os.path.normpath(line[5]).split(os.path.sep)
                line[5] = filepath_images + path[1].split(
                    '\\')[-1]
                validation_samples.append(line)

validation_samples_sequence = []
for idx, element in enumerate(validation_samples):
    # element = element[5:8]  # we grab only camera, speed and angle.
    if (idx <= ( len(validation_samples) - ((num_frames-1)*num_skip_frames+1) )):
        (validation_samples_sequence.append(validation_samples[idx:idx+((num_frames-1)*num_skip_frames+1):num_skip_frames]))

validation_samples_sequence = validation_samples_sequence[0:len(validation_samples_sequence):skip_line_valid]
validation_samples_sequence = sklearn.utils.shuffle(validation_samples_sequence)


print "Number of training samples: ", len(train_samples_sequence)
print "Number of validation samples: ", len(validation_samples_sequence)


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
                images_sequence = []
                for idx, frame in enumerate(batch_sample):

                    center_image = cv2.imread(frame[5])


                    images_sequence.append(center_image)
                    if idx == (num_frames-1):
                        # TODO: Scale accordingly
                        # Normalized output
                        # Angle: [-1, 1] --> [-1, 1]
                        angle = float(frame[6])
                        # Speed: [-10, 10] --> [-1, 1]
                        speed = float(frame[7]) / 0.3

                        outputs.append([angle, speed])

                    if aug:
                        pass

                images.append(images_sequence)

            x_train = np.array(images)
            y_train = np.array(outputs)
            if len(x_train.shape) < 5:
                x_train = np.expand_dims(x_train, axis=4)

            yield sklearn.utils.shuffle(x_train, y_train)


# compile and train the model using the generator function

train_generator = generator(train_samples_sequence, batch_size=batch_size_value, aug=0)
validation_generator = generator(
    validation_samples_sequence, batch_size=batch_size_value, aug=0)

# Load model for re-training or create a new one
if RE_TRAINING is True:
    model = models.load_model(filepath_model_load)
    if LOAD_WEIGHTS is True:
        model.load_weights(filepath_model_weights_load)

else:
	
    # TODO: Apply TimeDistributed NN.

    # keras NN model with functional API
    input_shape = [num_frames] + screen_size
    inputs = Input(shape=input_shape)

    ############ First Branch ############
    # Start Prespocessing
    x = layers.Lambda(lambda var: (var / 255.0) - 0.5)(inputs)  # Feature Normalization
    # x = layers.Lambda(lambda var: (var / 255.0))(inputs)  # Feature Normalization
    # x = layers.Lambda(lambda var: var[:, :, 185:355, :, :])(x)  # cropping
    x = layers.Lambda(lambda var: var[:, :, :, :, 0])(x)  # channel selection: BLUE
    x = layers.Permute((2, 3, 1))(x)
    x = layers.AveragePooling2D(pool_size=(4, 4))(x)  # pixelation
    # End Preprocessing

    # x = layers.SpatialDropout2D(0.1, dim_ordering='default')(x)

    x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    first_branch_output = layers.BatchNormalization()(x)

    # first_branch_output = layers.AveragePooling2D(pool_size=(2, 2))(x)

    ######################################

    ############ Second Branch ############
    # Start Prespocessing
    x = layers.Lambda(lambda var: (var / 255.0) - 0.5)(inputs)  # Feature Normalization
    # x = layers.Lambda(lambda var: (var / 255.0))(inputs)  # Feature Normalization
    # x = layers.Lambda(lambda var: var[:, :, 185:355, :, :])(x)  # cropping
    x = layers.Lambda(lambda var: var[:, :, :, :, 1])(x)  # channel selection: GREEN
    x = layers.Permute((2, 3, 1))(x)
    x = layers.AveragePooling2D(pool_size=(4, 4))(x)  # pixelation
    # End Preprocessing

    # x = layers.SpatialDropout2D(0.1, dim_ordering='default')(x)

    x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.maxnorm(4))(x)
    second_branch_output = layers.BatchNormalization()(x)

    # second_branch_output = layers.AveragePooling2D(pool_size=(2, 2))(x)

    ######################################

    x = layers.concatenate([first_branch_output, second_branch_output])

    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(1-0.75)(x)

    x = layers.Dense(1024, activation="relu", kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(1-0.5)(x)

    x = layers.Dense(512, activation="relu", kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(1-0.5)(x)

    x = layers.Dense(128, activation="relu", kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(1-0.5)(x)

    x = layers.Dense(64, activation="relu", kernel_constraint=constraints.maxnorm(4))(x)
    x = layers.BatchNormalization()(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    # Model compile
    model.compile(
        loss='mse',
        optimizer=optimizers.Nadam(lr=0.0002),  # 'adam' or 'SGD' or 'Nadam'
        metrics=['mse']
    )

# Model visualization
model.summary()
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# Chekpoints
callbacks = [
    # callbacks.EarlyStopping(patience=20),
    callbacks.ModelCheckpoint(filepath=filepath_weights_checkpoint, save_best_only=True, period=10),
    callbacks.TensorBoard(log_dir=filepath_tensorboard)   # to run tensorboard, execute in a terminal (in respective folder): tensorboard --logdir=./logs or tensorboard --logdir logs and open link in browser.
]

history = model.fit_generator(
    train_generator, steps_per_epoch=(len(train_samples_sequence) / batch_size_value), epochs=n_epoch, verbose=2,
    validation_data=validation_generator, validation_steps=(len(validation_samples_sequence) / batch_size_value),
    callbacks=callbacks,
    initial_epoch=initial_epoch,
    class_weight={0: 7, 1: 3}
)

# class_weight: le bajamos la importancia al  comando 'speed', dado que es el que mas se utiliza.
# Esto es para contrarrestar el data imbalance.
# Recordar: clases -> ('angle', 'speed')

# Save model
# model.save(filepath_model_save)

# with open('models/v2/driverless_model.json', 'w') as output_json:
#     output_json.write(model.to_json())
#
# # Save TensorFlow model
# tf.train.write_graph(
#     K.get_session().graph.as_graph_def(),
#     logdir='.',  # logdir='.'
#     name='models/v2/driverless_model.pb',
#     as_text=False)

print('Done training')
