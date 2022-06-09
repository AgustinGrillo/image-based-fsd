# load the prepared dataset
from numpy import load
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot
import cv2
import fileinput
import sys


# World cone randomizer
# for line in fileinput.input('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/track/track_3_mod.sdf', inplace=True):
#     if line.strip().startswith('<uri>'):
#         color = np.random.choice(['yellow', 'blue', 'orange_big'])
#         line = '      <uri>model://cone_' + color + '</uri>\n'
#     sys.stdout.write(line)


#### Code for compressing dataset ####
# data = load('dataset_images/dataset_compressed.npz')
# src_images, tar_images = data['arr_0'], data['arr_1']
# print('Loaded: ', src_images.shape, tar_images.shape)
# # plot source images
# n_samples = 3
# for i in range(n_samples):
# 	pyplot.subplot(2, n_samples, 1 + i)
# 	pyplot.axis('off')
# 	pyplot.imshow(src_images[i].astype('uint8'))
# # plot target image
# for i in range(n_samples):
# 	pyplot.subplot(2, n_samples, 1 + n_samples + i)
# 	pyplot.axis('off')
# 	pyplot.imshow(tar_images[i].astype('uint8'))
# pyplot.show()


#### CODE FOR RESIZING ####
# # load the dataset
# data = load('dataset_images/dataset_compressed.npz')
# src_images, tar_images = data['arr_0'], data['arr_1']
#
# cropped_img = src_images[10].astype('uint8')
# cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
#
# initial_shape = cropped_img.shape
# print("Initial cropped image size: ", initial_shape)
# cv2.imshow('Initial cropped', cropped_img)
# # cv2.waitKey(0)
#
# rescaled_img = cv2.resize(cropped_img, dsize=(512, 256))
# print("Enlarged image size: ", rescaled_img.shape)
# cv2.imshow('Enlarged', rescaled_img)
#
# restored_img = cv2.resize(rescaled_img, dsize=(640, 168))
# print("Restored image size: ", restored_img.shape)
# cv2.imshow('restored', restored_img)
# cv2.waitKey(0)

