import image_preprocessing
import cv2
import numpy as np
import os
import rospkg
import keras
from keras import Input, layers, models, callbacks, constraints
import tensorflow as tf
import math

layer_name = 'conv2d_2'
color_image = False

rospack = rospkg.RosPack()
path_imitation_learning = rospack.get_path('imitation_learning')
filepath_model_load = path_imitation_learning + "/models/v15/driverless_model_v15_2.h5"

img = cv2.imread('cone_test.jpg')

model = models.load_model(filepath_model_load)
model.summary()

img_expanded = np.expand_dims(img, axis=0)

pred = model.predict(img_expanded)

print pred

img_tensor = keras.backend.constant(img_expanded)

layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

out = feature_extractor(img_tensor)
out_np_o = keras.backend.eval(out)

size = np.shape(out_np_o)
filter = size[3]
img_height = size[1]
img_width = size[2]
n = int(math.sqrt(filter))
m = int(math.ceil(filter/n))

all_imgs = []
for i in range(filter):
    if not color_image:
        # print i
        img = out_np_o[0, :, :, i:i + 1]
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        all_imgs.append(img)
    else:
        img = out_np_o[0, :, :, :]
        img = img.astype(np.uint8)
        all_imgs.append(img)
        filter = 1
        n = 1
        m = 1

# Build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5

width = n * img_width + (n - 1) * margin  # columnas
height = m * img_height + (m - 1) * margin  # filas
stitched_filters = np.zeros((height, width, 3))

# Fill the picture with our saved filters
for i in range(m):
    for j in range(n):
        if i * n + j < filter:
            img = all_imgs[i * n + j]
            # print (i * n + j)
            stitched_filters[
            (img_height + margin) * i: (img_height + margin) * i
                                       + img_height,
            (img_width + margin) * j: (img_width + margin) * j + img_width,
            :] = img

keras.preprocessing.image.save_img("activation_layer_2.png", stitched_filters[:, :, [2, 1, 0]])
