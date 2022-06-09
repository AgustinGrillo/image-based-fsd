import image_preprocessing
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import rospkg
import keras
import tensorflow as tf


##################################################################################################################

img = cv2.imread('test.jpg')
cv2.imshow('img_raw', img)

img_b = img[:, :, 0]
cv2.imshow('img_blue', img_b)

img_g = img[:, :, 1]
cv2.imshow('img_green', img_g)

img_r = img[:, :, 2]
cv2.imshow('img_red', img_r)

# plt.show()
while cv2.waitKey(0) != 27:
    pass


############ ALIMENTAICON DE SECUENCIA DE IMAGENES A LA RED ###########################################

# def channel_getter(img):
#     import tensorflow as tfff  # importamos tensorflow aca porque sino crashea cuando carga el modelo
#     new_img = tfff.expand_dims(img[:, :, :, :, 2], axis=4)
#     return new_img
#
#
# img = cv2.imread('cones_test.jpg')
# img_orig = np.expand_dims(img, axis=0)
# img = img_orig
#
#
# for i in range(5):
#     img = np.append(img, img_orig, axis=0)
#     print(img.shape)
#
# # cv2.imshow('original image', img)
# print("Tamano Original:",  img.shape)
#
#
# img_expanded = np.expand_dims(img, axis = 0)
# # img_expanded = np.expand_dims(img_expanded, axis = 3)
#
# img_tensor = keras.backend.constant(img_expanded)
# # x_test = keras.layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=(480, 640, 3))( tf.expand_dims(img_tensor[:,:,:,0], axis=3) )
# # x_test = keras.layers.AveragePooling2D(pool_size=(4, 4))(x_test)
# # img_norm = keras.backend.eval(x_test)
#
#
#
# # x_test = keras.layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=(6, 480, 640, 3))( img_tensor )
# x_test = keras.layers.Lambda(lambda var: var[:, :, 185:355, :, :])(img_tensor)
# x_test = keras.layers.Lambda(lambda var: var[:, :, :, :, 0])(x_test)
# x_test = keras.layers.Permute((2, 3, 1))(x_test)
# # x_test = keras.layers.Lambda(lambda var: var[:, :, :, :, 0])(x_test)
# x_test = keras.layers.AveragePooling2D(pool_size=(4, 4))(x_test)
# img_norm = keras.backend.eval(x_test)
#
# img_rec = img_norm[0, :, :, :]
# img_rec = img_rec.astype(np.uint8)
# print("Tamano Final:",  img_rec.shape)
# # cv2.imshow('mod image con keras', img_rec)
#
# # cv2.waitKey(0)


############ REEMPLAZO DE IMAGE PREPROCESSING DENTRO DEL MODELO ###########################################
# ip = image_preprocessing.process()
# # original image
# img = ip.load_image('cones_test.jpg')
#
#
#
# cv2.imshow('original image', img)
# print("Tamano Original:",  img.shape)
#
#
# img_expanded = np.expand_dims(img, axis = 0)
# # img_expanded = np.expand_dims(img_expanded, axis = 3)
#
# img_tensor = keras.backend.constant(img_expanded)
# # x_test = keras.layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=(480, 640, 3))( tf.expand_dims(img_tensor[:,:,:,0], axis=3) )
# # x_test = keras.layers.AveragePooling2D(pool_size=(4, 4))(x_test)
# # img_norm = keras.backend.eval(x_test)
#
# x_test = keras.layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=(480, 640, 3))( img_tensor )
# x_test = keras.layers.AveragePooling2D(pool_size=(4, 4))(x_test)
# x_test = keras.layers.Lambda(lambda var: tf.expand_dims(var[:,:,:,0], axis=3))(x_test)  # Feature Normalization
# img_norm = keras.backend.eval(x_test)
#
# img_rec = img_norm[0, :, :, :]
# img_rec = img_rec.astype(np.uint8)
# print("Tamano Final:",  img_rec.shape)
# cv2.imshow('mod image con keras', img_rec)
#
# cv2.waitKey(0)


##################################################################################################################


# ip = image_preprocessing.process()
#
#
# # original image
# img = ip.load_image('cones_test.jpg')
#
# processed_image = img[:, :, 0]  # blue channel
# processed_image = processed_image[185:185 + 170, :]  # crop: [185:355, :]
# ip.show_window('cropped', processed_image)
# print('cropped size:', processed_image.shape)
#
# # test funciton
# test, test_scaled = ip.process_img(img, pixelation=4, screen_size=[170, 640])
# ip.show_window('test', test)
# print('test size:', test.shape)
# ip.show_window('test scaled', test_scaled)
# print('test scaled size:', test_scaled.shape)
#
# # for soft ending
# ip.destroy_windows()




###################################### TO CHECK POOLING #####################################3
# ip = image_preprocessing.process()
# # original image
# img = ip.load_image('cones_test.jpg')
# img = img[:, :, 0]
# img = img[185:185 + 170, :]
#
# cv2.imshow('original image', img)
# print("Tamano Original:",  img.shape)
#
# img_prepro, img_rescaled = ip.process_img(ip.load_image('cones_test.jpg'), 4)
# cv2.imshow('mod con pixelado', img_prepro)
# print("Tamano pixerlado:",  img_prepro.shape)
#
# img_expanded = np.expand_dims(img, axis = 0)
# img_expanded = np.expand_dims(img_expanded, axis = 3)
#
# img_tensor = keras.backend.constant(img_expanded)
# x_test = keras.layers.AveragePooling2D(pool_size=(4, 4))(img_tensor)
# # x_test = keras.layers.MaxPooling2D(pool_size=(2, 2))(x_test)
# img_norm = keras.backend.eval(x_test)
#
# img_rec = img_norm[0, :, :, 0]
# img_rec = img_rec.astype(np.uint8)
# print("Tamano Final:",  img_rec.shape)
# cv2.imshow('mod image con keras', img_rec)
#
# cv2.waitKey(0)
############################################################################################





