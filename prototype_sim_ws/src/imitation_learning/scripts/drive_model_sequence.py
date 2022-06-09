#!/usr/bin/env python

# SHEBANG original: !/usr/bin/env python
# Modificar shebang en funcion de necesidades.
# Si se tiene los paquetes en el root, entonces deberia andar con el original.
# Si se tiene los paquetes en algun enviroment, buscar el binario de python dentro del mismo (interpreter(?)), y modificar el shebang por su path.
# Ejemplo: #!/home/agus/miniconda3/envs/fss/bin/python


import rospy
import time
import sys
import base64
from datetime import datetime
import os
# import shutil/fss_sim
import numpy as np
import cv2
import image_preprocessing
from keras import models
import rospkg
import keras
import tensorflow as tf

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import json
from keras.models import model_from_json, load_model
import h5py
from keras import __version__ as keras_version

DEBUG = False  # To visualize the processed camera input and commands

path_model = "/weights/sequence/vS0/weights-improvement-vS0.0-70-0.04.hdf5"

layer_name = "cropping2d_1"


# def shift(xs, n):
  #  xs.append(nan)
#return e

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

class cmd_vel_node(object):
    def __init__(self):

        # Setup Paths
        rospack = rospkg.RosPack()
        path_imitation_learning = rospack.get_path('imitation_learning')
        self.model_path = path_imitation_learning + path_model
        print 'Loaded Model Path:', self.model_path

        """ Variables """
        self.cmdvel = Twist()
        self.bridge = CvBridge()
        self.num_frames = 3
        self.skip_frames = 3
        self.size_buffer = (self.num_frames - 1) * self.skip_frames + 1
        self.buffer_full = False
        self.count_img = 0

        self.latest_image = None
        self.preprocessed_image = None
        self.imgRcvd = False
        self.buffer = np.zeros((self.size_buffer, 360, 640, 3)).astype(np.uint8)  # 16

        """ROS Publications """
        self.cmdVel_pub = rospy.Publisher("/fs/cmd_vel", Twist, queue_size=1)

        if DEBUG is True:
            self.debug_pub = rospy.Publisher("/image_converter/debug_video", Image, queue_size=10)

        """ROS Subscriptions """
        self.image_sub = rospy.Subscriber("/fs/c1/image_raw", Image, self.cvt_image)

        self.im_pre = image_preprocessing.process()

        # self.outputImage = None
        # self.debugImage = None

    def cvt_image(self, data):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # convert ros image to opencv image

        except CvBridgeError as e:
            print(e)

        # shift(self.buffer, -1)
        # self.buffer[self.size_buffer-1] = self.latest_image

        self.buffer = np.append(self.buffer, np.expand_dims(self.latest_image, axis=0), axis=0)
        self.buffer = self.buffer[1:]

        if self.count_img < self.size_buffer:
            self.count_img = self.count_img + 1

            if self.count_img == self.size_buffer:
                self.buffer_full = True

        if self.imgRcvd != True:
            self.imgRcvd = True

    def run(self):

        # Load model
        model = load_model(self.model_path)
        print("Model loaded")

        model.summary()

        # Set up a model that returns the activation values for our target layer
        if DEBUG is True:
            layer = model.get_layer(name=layer_name)
            feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

        while not rospy.is_shutdown():
            # Only run loop if we have an image
            if self.imgRcvd and self.buffer_full:

                # preprocessed_image, preprocessed_image_scaled = self.im_pre.process_img(self.latest_image, pixelation,
                #                                                                        screen_size)
                # preprocessed_image = np.expand_dims(self.buffer[0:self.skip_frames:(self.size_buffer)], axis=0)  # we add batch dimension
                # preprocessed_image = np.expand_dims(preprocessed_image, axis=3)  # we add color dimension (if grayscale)

                preprocessed_image = self.buffer[0::self.skip_frames]    # self.buffer[self.size_buffer::-self.skip_frames]
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # we add batch dimension


                # Prediction of Neural Network based on image captured
                # pred[0]=angle,  pred[1]=speed
                pred = model.predict(preprocessed_image)

                # Publish
                self.cmdvel.angular.z = pred[0][0]
                # TODO: Scale accordingly
                self.cmdvel.linear.x = pred[0][1] * 0.3
                # self.cmdvel.linear.x = 2 if self.cmdvel.linear.x > 2 else self.cmdvel.linear.x
                self.cmdVel_pub.publish(self.cmdvel)
                self.imgRcvd = False

                print 'Speed:', "%.2f" % self.cmdvel.linear.x, 'Angle:', "%.2f" % self.cmdvel.angular.z

                if DEBUG is True:
                    img_tensor = keras.backend.constant(preprocessed_image)
                    out = feature_extractor(img_tensor)
                    out_np = keras.backend.eval(out)
                    out_np_o = out_np.astype(np.uint8)

                    out_np_o = out_np_o[0, :, :, :]
                    # out_np_o = np.expand_dims(out_np_o, axis=2)
                    # out_np_o = out_np_o * (255 / out_np_o.max())

                    imgmsg = self.bridge.cv2_to_imgmsg(out_np_o, "bgr8")

                    self.debug_pub.publish(imgmsg)

                    # self.im_pre.show_window('processed image scaled', preprocessed_image_scaled)
                    # cv2.waitKey(1)


def main(args):
    rospy.init_node('driver_node', anonymous=False)

    cmd = cmd_vel_node()

    cmd.run()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
