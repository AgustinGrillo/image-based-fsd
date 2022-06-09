#!/usr/bin/env python
"""
Title: Visualizing what convnets learn
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/29
Last modified: 2020/05/29
Description: Displaying the visual patterns that convnet filters respond to.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from keras import Input, layers, models
import rospkg
# import keras
# from keras.models import model_from_json, load_model


# The dimensions of our input image
layer_name = "conv_1"
display_all = False
img_width = 360
img_height = 640
img_depth = 3
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
path_model = "/weights/v0/weights-improvement-v0.0-40-0.04.hdf5"


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

"""
## Introduction
In this example, we look into what sort of visual patterns image classification models
learn. We'll be using the `ResNet50V2` model, trained on the ImageNet dataset.
Our process is simple: we will create input images that maximize the activation of
specific filters in a target layer (picked somewhere in the middle of the model: layer
`conv3_block4_out`). Such images represent a visualization of the
pattern that the filter responds to.
"""

"""
## Setup
"""

rospack = rospkg.RosPack()
path_imitation_learning = rospack.get_path('imitation_learning')
model_path = path_imitation_learning + path_model  # Paths to modify!!

print 'Loaded Model Path:', model_path

"""
## Build a feature extraction model
"""

# Build a ResNet50V2 model loaded with pre-trained ImageNet weights
# model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)

# model = load_model(model_path)
model = keras.models.load_model(model_path)
model.summary()

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

"""
## Set up the gradient ascent process
The "loss" we will maximize is simply the mean of the activation of a specific filter in
our target layer. To avoid border effects, we exclude border pixels.
"""


def compute_loss(input_image, filter_index):
    input_image_rescaled = (input_image + 0.5) * 255
    activation = feature_extractor(input_image_rescaled)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


"""
Our gradient ascent function simply computes the gradients of the loss above
with regard to the input image, and update the update image so as to move it
towards a state that will activate the target filter more strongly.
"""


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
        # Compute gradients.
        grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    img = tf.clip_by_value(img, -1.0, 1.0)
    return loss, img


"""
## Set up the end-to-end filter visualization loop
Our process is as follow:
- Start from a random image that is close to "all gray" (i.e. visually netural)
- Repeatedly apply the gradient ascent step function defined above
- Convert the resulting input image back to a displayable form, by normalizing it,
center-cropping it, and restricting it to the [0, 255] range.
"""


def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, img_depth))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 200
    learning_rate = 100.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
        # print('Epoch: {}         Loss: {}'.format(iteration, keras.backend.eval(loss)))

    # Decode the resulting input image
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result_output = sess.run(img[0])

    img = deprocess_image(result_output)

    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


"""
Let's try it out with filter 0 in the target layer:
"""

if not display_all:
    dummy_img = keras.backend.eval(feature_extractor(initialize_image()))
    num_filters = dummy_img.shape[3]
    filter_idx = np.random.randint(num_filters)

    loss, img = visualize_filter(filter_idx)
    keras.preprocessing.image.save_img("stiched_filters.png", img)
    # plt.imshow(img)
    # plt.show()


"""
## Visualize the first 64 filters in the target layer
Now, let's make a 8x8 grid of the first 64 filters
in the target layer to get of feel for the range
of different visual patterns that the model has learned.
"""

if display_all:
    # Compute image inputs that maximize per-filter activations
    dummy_img = keras.backend.eval(feature_extractor(initialize_image()))
    num_filters = dummy_img.shape[3]
    n = int(np.floor(np.sqrt(num_filters)))
    m = int(np.ceil(float(num_filters)/float(n)))

    all_imgs = []
    # for filter_index in range(num_filters):
    for filter_index in range(num_filters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5

    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = m * cropped_width + (m - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, img_depth))

    # Fill the picture with our saved filters
    for i_filter in range(num_filters):
        img = all_imgs[i_filter]
        i = int(np.floor(float(i_filter) / float(n)))
        j = int(i_filter - i * n)
        stitched_filters[
        (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
        (cropped_height + margin) * j: (cropped_height + margin) * j + cropped_height,
        :] = img
    keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)

    # from IPython.display import Image, display
    #
    # display(Image("stiched_filters.png"))

"""
Image classification models see the world by decomposing their inputs over a "vector
basis" of texture filters such as these.
See also
[this old blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
for analysis and interpretation.
"""
