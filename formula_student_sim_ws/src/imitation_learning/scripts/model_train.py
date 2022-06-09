#!/usr/bin/env python


import os
import csv
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input, layers, models, callbacks
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
from model_def import build_model_v15 as model_sel
import rospkg

rospack = rospkg.RosPack()
path_imitation_learning = rospack.get_path('imitation_learning')

### Variables to modify ###
RE_TRAINING = False
LOAD_WEIGHTS = False

batch_size_value = 16  # 32 64 1024
n_epoch = 100  # 10
initial_epoch = 0

version = "15"
subversion = "2"

filepath_model_load = path_imitation_learning + "/models/v5/driverless_model_v5_0.h5"  # 'models/v2/driverless_model.h5'
filepath_model_weights_load = path_imitation_learning + "/weights/v2/weights-improvement-v2.5-29-0.72.hdf5"  # '/home/agus/ros/fss_ws/src/imitation_learning/weights/v2/weights-improvement-v2.4-07-1.24.hdf5'

data_training = ["/output/data_track_5", "/output/data_track_2", "/output/data_track_9", "/output/data_track_11",
                 "/output/data_track_8"]

data_validation = ["/output/data_track_12", "/output/data_track_10"]

skip_line_train = 2
skip_line_valid = 10  # Cuantos frames me salteo

screen_size = [480, 640, 3]

###########################

filepath_model_save = path_imitation_learning + "/models/v" + version + "/driverless_model_v" + version + "_" + subversion + ".h5"
filepath_tensorboard = path_imitation_learning + "/logs/run" + version + "." + subversion
filepath_weights_checkpoint = path_imitation_learning + "/weights/v" + version + "/weights-improvement-v" + version + "_" + subversion + "-{epoch:02d}-{val_loss:.2f}.hdf5"

print "File Path Model:", filepath_model_save

### GRAB TRAINING DATA ####

train_samples = []
for data_name in data_training:

    filepath_data = path_imitation_learning + data_name + "/interpolated.csv"  # '/home/agus/ros/fss_ws/src/imitation_learning/output/data_1/interpolated.csv'
    filepath_images = path_imitation_learning + data_name + "/center/"  # '/home/agus/ros/fss_ws/src/imitation_learning/output/data_1/center/'

    print "Training Paths:", filepath_data

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

    print "Validation Paths:", filepath_data

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

print "Number of training samples: ", len(train_samples)
print "Number of validation samples: ", len(validation_samples)


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

                    # image preprocessing
                    # processed_image, processed_image_scaled = im_pre.process_img(center_image, pixelation, screen_size[:2])
                    angle = float(batch_sample[6])
                    speed = float(batch_sample[7])
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
    # keras NN model with functional API
    model = model_sel(screen_size)
    # Model compile
    model.compile(
        loss='mse',
        optimizer='adam')  # 'adam' or 'SGD'

# Model visualization
model.summary()
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# Chekpoints
callbacks = [
    callbacks.EarlyStopping(patience=7)
    # callbacks.ModelCheckpoint(filepath=filepath_weights_checkpoint, save_best_only=True, period=1),
    # callbacks.TensorBoard(log_dir=filepath_tensorboard)   # to run tensorboard, execute in a terminal (in respective folder): tensorboard --logdir=./logs or tensorboard --logdir logs and open link in browser.
]

history = model.fit_generator(
    train_generator, steps_per_epoch=(len(train_samples) / batch_size_value), epochs=n_epoch, verbose=1,
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
#     name='models/v2/driverless_model.pb',
#     as_text=False)

print('Done training')

#### PRINT OUT SUMMARY OF TRAINING ####

path_split = filepath_model_save.split('/')

path_txt = ""

for i in range(1, len(path_split)):
    path_txt = path_txt + '/' + path_split[i]
    if path_split[i - 1] == "models":
        model_name = path_split[i + 1]
        break

print path_txt

f = open(path_txt + "/README.txt", "a+")

f.write("-------------------------------------------------------- \n\n")

f.write("Model Name: " + model_name + "\n\n")

f.write("Trained in: \n")

for data_name in data_training:
    f.write("   -" + data_name + "\n")

f.write("Validated  in: \n")

for data_name in data_validation:
    f.write("   -" + data_name + "\n")

f.write("\n Epochs:  %d   \n" % n_epoch)
f.write("\n Batch Sample Size: %d \n " % batch_size_value)

f.write("\n\n\n")

f.close()

# model = Sequential()
#
# # trim image to only see section with road
# model.add(Cropping2D(cropping=((184, 133), (0, 0)), input_shape=(480, 640, 3)))
#
# # Preprocess incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#
# #Nvidia model
# model.add(
#     Convolution2D(
#         24, (5, 5), activation="relu", name="conv_1", strides=(2, 2)))
# model.add(
#     Convolution2D(
#         36, (5, 5), activation="relu", name="conv_2", strides=(2, 2)))
# model.add(
#     Convolution2D(
#         48, (5, 5), activation="relu", name="conv_3", strides=(2, 2)))
# model.add(SpatialDropout2D(.5, dim_ordering='default'))
#
# model.add(
#     Convolution2D(
#         64, (3, 3), activation="relu", name="conv_4", strides=(1, 1)))
# model.add(
#     Convolution2D(
#         64, (3, 3), activation="relu", name="conv_5", strides=(1, 1)))
#
# model.add(Flatten())
#
# model.add(Dense(1164))
# model.add(Dropout(.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(1))
#
# model.compile(loss='mse', optimizer='adam')
# model.summary()
#
# # checkpoint
# filepath = "/home/agus/ros/fss_ws/src/imitation_learning/weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
# checkpoint = ModelCheckpoint(
#     filepath,
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True,
#     mode='auto',
#     period=1)
# callbacks_list = [checkpoint]
#
# # Fit the model
# history_object = model.fit_generator(
#     train_generator,
#     steps_per_epoch=(len(train_samples) / batch_size_value),
#     validation_data=validation_generator,
#     validation_steps=(len(validation_samples) / batch_size_value),
#     callbacks=callbacks_list,
#     epochs=n_epoch)
#
# # Save model
# model.save('model.h5')
# with open('model.json', 'w') as output_json:
#     output_json.write(model.to_json())
#
# # Save TensorFlow model
# tf.train.write_graph(
#     K.get_session().graph.as_graph_def(),
#     logdir='.',
#     name='model.pb',
#     as_text=False)
#
# # Plot the training and validation loss for each epoch
# print('Generating loss chart...')
# '''
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('model.png')
# '''
#
# # Done
# print('Done.')
