#!/usr/bin/env python

# Este codigo entrena una secuencia de imagenes con Neural Circuit Policies.
# idealmente una secuencia de 6 (salteandose cada 3 a 10 FPS). Suponiendo una velocidad de crucero de 15 m/s del formula.
# Se deben poder ver conos anteriores.

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import Input, layers, models, callbacks, constraints
# from keras import optimizers
# from keras import Input, layers, models, callbacks, constraints
import kerasncp as kncp

import sklearn
from sklearn.model_selection import train_test_split
import rospkg


rospack = rospkg.RosPack()
path_imitation_learning = rospack.get_path('imitation_learning')

### Variables to modify ###
RE_TRAINING = False
LOAD_WEIGHTS = False

batch_size_value = 16  # 32 64 1024
n_epoch = 1000  # 10
initial_epoch = 0

version = "NCP1"
subversion = "0"

# Number of frames to feed the sequence, and how many frames to skip.
# (if num_skip_frames=1, then the frames are consecutive)
num_frames = 6
num_skip_frames = 3
######################################

filepath_model_load = path_imitation_learning + "/models/sequence/vS1/driverless_model.h5"
filepath_model_weights_load = path_imitation_learning + "/weights/sequence/vS1/weights-improvement-v2.5-29-0.72.hdf5"

# data_training = ["/output/data_track_5", "/output/data_track_2", "/output/data_track_9", "/output/data_track_11",
#                  "/output/data_track_8"]

# data_validation = ["/output/data_track_12", "/output/data_track_1"]

# data_training = ["/output/data_train"]
# data_validation = ["/output/data_train"]

data_training = ["/output/data_track_5", "/output/data_track_6"]
data_validation = ["/output/data_track_9", "/output/data_track_10"]

skip_line_train = 30  # 4
skip_line_valid = 30  # 10  # Cuantos frames me salteo

screen_size = [480, 640, 3]

###########################

filepath_model_save = path_imitation_learning + "/models/RNN-NCP/v" + version + "/driverless_model_v" + version + "." + subversion + ".h5"
filepath_tensorboard = path_imitation_learning + "/logs/run" + version + "." + subversion
filepath_weights_checkpoint = path_imitation_learning + "/weights/RNN-NCP/v" + version + "/weights-improvement-v" + version + "." + subversion + "-{epoch:02d}-{val_loss:.2f}.hdf5"

print("File Path Model:", filepath_model_save)

### GRAB TRAINING DATA ####

train_samples = []
for data_name in data_training:

    filepath_data = path_imitation_learning + data_name + "/interpolated.csv"  # '/home/agus/ros/fss_ws/src/imitation_learning/output/data_1/interpolated.csv'
    filepath_images = path_imitation_learning + data_name + "/center/"  # '/home/agus/ros/fss_ws/src/imitation_learning/output/data_1/center/'

    print("Training Paths:", filepath_data)

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

    print("Validation Paths:", filepath_data)

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


print("Number of training samples: ", len(train_samples_sequence))
print("Number of validation samples: ", len(validation_samples_sequence))


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
                outputs_sequence = []
                for idx, frame in enumerate(batch_sample):

                    center_image = cv2.imread(frame[5])


                    images_sequence.append(center_image)

                    angle = float(frame[6])
                    speed = float(frame[7])
                    # outputs.append([angle, speed])

                    outputs_sequence.append([angle, speed])

                    if aug:
                        pass

                images.append(images_sequence)
                outputs.append(outputs_sequence)

            x_train = np.array(images)
            y_train = np.array(outputs)
            if len(x_train.shape) < 5:
                x_train = np.expand_dims(x_train, axis=0)

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

    ncp_wiring = kncp.wirings.NCP(
        inter_neurons=20,  # Number of inter neurons
        command_neurons=10,  # Number of command neurons
        motor_neurons=2,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=5,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=6,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=4,  # How many incomming syanpses has each motor neuron
    )
    ncp_cell = kncp.LTCCell(
        ncp_wiring,
        initialization_ranges={
            # Overwrite some of the initalization ranges
            "w": (0.2, 2.0),
        },
    )

    # keras NN model with functional API
    # input_shape = [None] + screen_size
    inputs = Input(shape=(None, screen_size[0], screen_size[1], screen_size[2]))

    # Start Prespocessing
    x = layers.Lambda(lambda var: (var / 255.0))(inputs)  # Feature Normalization
    x = layers.Lambda(lambda var: var[:, :, 185:355, :, :])(x)  # cropping
    #x = layers.Lambda(lambda var: var[:, :, :, :, [0, 2]])(x)  # channel selection: BLUE and RED
    x = layers.TimeDistributed(layers.AveragePooling2D(pool_size=(4, 4)))(x)
    # End Preprocessing

    x = layers.TimeDistributed(layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4)))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.TimeDistributed(layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4)))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.TimeDistributed(layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4)))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.TimeDistributed(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4)))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.TimeDistributed(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4)))(x)
    # x = layers.BatchNormalization()(x)

    """
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu",
                      kernel_constraint=constraints.MaxNorm(4))(x)
    x = layers.BatchNormalization()(x)
    """

    # Flatten layers
    x = layers.TimeDistributed(layers.Flatten())(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(1-0.75)(x)

    x = layers.TimeDistributed(layers.Dense(1024, activation="relu", kernel_constraint=constraints.MaxNorm(4)))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(1-0.5)(x)

    outputs = layers.RNN(ncp_cell, return_sequences=True)(x)
    #outputs = layers.TimeDistributed(layers.Activation("linear"))(x)  # output

    # outputs = layers.TimeDistributed(layers.Dense(2, activation="linear", kernel_constraint=constraints.MaxNorm(4)))(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    # Model compile
    model.compile(
        loss='mse',
        optimizer=optimizers.Nadam(lr=0.0005),  # 'adam' or 'SGD' or 'Nadam'
        metrics=['mse']
    )

# Model visualization
model.summary()
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# Chekpoints
callbacks = [
    # callbacks.EarlyStopping(patience=20),
    # callbacks.ModelCheckpoint(filepath=filepath_weights_checkpoint, save_best_only=True, period=5),
    callbacks.TensorBoard(log_dir=filepath_tensorboard)   # to run tensorboard, execute in a terminal (in respective folder): tensorboard --logdir=./logs or tensorboard --logdir logs and open link in browser.
]

history = model.fit(
    train_generator, steps_per_epoch=(len(train_samples_sequence) / batch_size_value), epochs=n_epoch, verbose=1,
    validation_data=validation_generator, validation_steps=(len(validation_samples_sequence) / batch_size_value),
    callbacks=callbacks,
    initial_epoch=initial_epoch,
    class_weight={0: 7, 1: 3}
)
# VER QUE EL MODEL.FIT NO ACEPTA GENERATORS EN EL VALIDATION. (VER KERAS API)

# class_weight: le bajamos la importancia al  comando 'speed', dado que es el que mas se utiliza.
# Esto es para contrarrestar el data imbalance.
# Recordar: clases -> ('angle', 'speed')

# Save model (uncoment for saving!!!!!!!!!!!!!!!!!!!!!!!!)
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

