#!/usr/bin/env python


import os
import csv
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input, layers, models, callbacks
from keras import backend as K
# from keras.models import Model, Sequential
# from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Lambda, Cropping2D
# from keras.layers.convolutional import Convolution2D
# from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split


RE_TRAINING = True

# NOTES:
# MANUAL MODIFICATIONS (TO AUTOMATE IN THE FUTURE):
# To re-train a NN with more data, select RE-TRAINING = True.
# Modify or verify paths for:
# - interpolated.csv
# - model to load (if re-training) and specific weight (if needed).
# - weights and tensorboard logs in callback

samples = []
with open('/home/agus/ros/fa_ws/src/imitation_learning/output/interpolated.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

sklearn.utils.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Number of traing samples: ", len(train_samples))
print("Number of validation samples: ", len(validation_samples))

screen_size = [165, 640, 1]  # grayscale image

#index,timestamp,width,height,frame_id,filename,angle,speed
def generator(samples, batch_size=32, aug=0):   # aug: augmentation
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            #print(batch_samples)
            images = []
            outputs = []
            for batch_sample in batch_samples:
                if batch_sample[5] != "filename":
                    path = os.path.normpath(batch_sample[5]).split(os.path.sep)
                    name = '/home/agus/ros/fa_ws/src/imitation_learning/output/center/' + path[1].split(
                        '\\')[-1]
                    center_image = cv2.imread(name)
                    #center_image = cv2.resize(
                    #    center_image,
                    #    (320, 180))  #resize from 720x1280 to 180x320
                    #plt.imshow(left_image)
                    #plt.show()
                    # image preprocessing
                    processed_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2GRAY)
                    processed_image = processed_image[190:190+screen_size[0], :]   # [190:355, :]
                    angle = float(batch_sample[6])
                    speed = float(batch_sample[7])
                    images.append(processed_image)
                    outputs.append([angle, speed])

                    if aug:
                        flip_image = np.fliplr(processed_image)
                        flip_angle = -1 * angle
                        images.append(flip_image)
                        outputs.append([flip_angle, speed])

            x_train = np.array(images)
            y_train = np.array(outputs)
            if len(x_train.shape) < 4:
                x_train = np.expand_dims(x_train, axis=3)


            yield sklearn.utils.shuffle(x_train, y_train)


# compile and train the model using the generator function
batch_size_value = 32
n_epoch = 10

train_generator = generator(train_samples, batch_size=batch_size_value, aug=1)
validation_generator = generator(
    validation_samples, batch_size=batch_size_value, aug=0)


# Load model for re-training or create a new one
if RE_TRAINING is True:
    model = models.load_model('models/v1/driverless_model.h5')
    model.load_weights('/home/agus/ros/fa_ws/src/imitation_learning/weights/v1/weights-improvement-v1.0-08-0.80.hdf5')

else:
    # keras NN model with functional API
    inputs = Input(shape=screen_size)
    x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    # Model compile
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mse']
    )


# Model visualization
model.summary()
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# Chekpoints
filepath = "/home/agus/ros/fa_ws/src/imitation_learning/weights/v1/weights-improvement-v1.2-{epoch:02d}-{val_loss:.2f}.hdf5"
callbacks = [
    callbacks.EarlyStopping(patience=2),
    callbacks.ModelCheckpoint(filepath=filepath, save_best_only=True, period=1),
    callbacks.TensorBoard(log_dir='./logs/run_2')   # to run tensorboard, execute in a terminal: tensorboard --logdir=./logs or tensorboard --logdir logs and open link in browser.
]

history = model.fit_generator(
    train_generator, steps_per_epoch=(len(train_samples) / batch_size_value), epochs=n_epoch, verbose=1,
    validation_data=validation_generator, validation_steps=(len(validation_samples) / batch_size_value),
    callbacks=callbacks,
    class_weight={0: 7, 1: 3},
    initial_epoch=0  # modify it!!
)
# class_weight: le bajamos la importancia al  comando 'speed', dado que es el que mas se utiliza.
# Esto es para contrarrestar el data imbalance.
# Recordar: clases -> ('angle', 'speed')

# Save model
model.save('models/v1/driverless_model.h5')

with open('models/v1/driverless_model.json', 'w') as output_json:
    output_json.write(model.to_json())

# Save TensorFlow model
tf.train.write_graph(
    K.get_session().graph.as_graph_def(),
    logdir='.',  # logdir='.'
    name='models/v1/driverless_model.pb',
    as_text=False)

print('Done training')








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
