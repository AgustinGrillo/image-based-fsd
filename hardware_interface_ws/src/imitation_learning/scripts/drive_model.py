#!/usr/bin/env python

# SHEBANG original: !/usr/bin/env python
# Modificar shebang en funcion de necesidades.
# Si se tiene los paquetes en el root, entonces deberia andar con el original.
# Si se tiene los paquetes en algun enviroment, buscar el binario de python dentro del mismo (interpreter(?)), y modificar el shebang por su path.
# Ejemplo: #!/home/agus/miniconda3/envs/fss/bin/python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import rospy
import time
import sys
import base64
from datetime import datetime
# import shutil/fss_sim
import numpy as np
import cv2
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

path_model = "/weights/v1/weights-improvement-v1.0-100-0.00.hdf5"

layer_name = "conv_2"

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

        self.latest_image = None
        self.preprocessed_image = None
        self.imgRcvd = False

        """ROS Publications """
        self.cmdVel_pub = rospy.Publisher("/fs/cmd_vel", Twist, queue_size=1)

        if DEBUG is True:
            self.debug_pub = rospy.Publisher("/image_converter/debug_video", Image, queue_size=10)

        """ROS Subscriptions """
        self.image_sub = rospy.Subscriber("/csi_cam_0/image_rect_color", Image, self.cvt_image)

        # self.outputImage = None
        # self.debugImage = None

    def cvt_image(self, data):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # convert ros image to opencv image
        except CvBridgeError as e:
            print(e)
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
            if self.imgRcvd:

                preprocessed_image = np.expand_dims(self.latest_image, axis=0)  # we add batch dimension

                # Prediction of Neural Network based on image captured
                # pred[0]=angle,  pred[1]=speed
                pred = model.predict(preprocessed_image)

                # Publish
                self.cmdvel.angular.z = pred[0][0]
                # TODO: Scale accordingly
                self.cmdvel.linear.x = pred[0][1] * 0.3 * 0.5
                # self.cmdvel.linear.x = 2 if self.cmdvel.linear.x > 2 else self.cmdvel.linear.x
                self.cmdVel_pub.publish(self.cmdvel)
                self.imgRcvd = False

                print 'Speed:', "%.2f" % self.cmdvel.linear.x, 'Angle:', "%.2f" % self.cmdvel.angular.z

                if DEBUG is True:
                    img_tensor = keras.backend.constant(preprocessed_image)
                    out = feature_extractor(img_tensor)
                    out_np = keras.backend.eval(out)
                    out_np_o = out_np[0, :, :, :]

                    height = out_np_o.shape[0]
                    width = out_np_o.shape[1]
                    num_filters = out_np_o.shape[-1]
                    separator_pixels = 1

                    cols = int(np.floor(np.sqrt(num_filters)))
                    rows = int(np.ceil(float(num_filters)/float(cols)))
                    concatenated_array = np.ones([rows*height + (rows-1)*separator_pixels, cols*width + (cols-1)*separator_pixels]) * 255.0

                    for i_filter in range(num_filters):
                        filter_array = out_np_o[:, :, i_filter]
                        filter_array = filter_array * (255 / filter_array.max())
                        concatenated_row = int(np.floor(float(i_filter)/float(cols)))
                        concatenated_col = int(i_filter - concatenated_row * cols)
                        concatenated_array[concatenated_row * (height+separator_pixels):concatenated_row * (height+separator_pixels) + height, concatenated_col * (width+separator_pixels):concatenated_col * (width+separator_pixels) + width] = filter_array

                    concatenated_array = concatenated_array.astype(np.uint8)

                    imgmsg = self.bridge.cv2_to_imgmsg(concatenated_array)

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
