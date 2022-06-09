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

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import json
from keras.models import model_from_json, load_model
import h5py
from keras import __version__ as keras_version

DEBUG = False  # To visualize the processed camera input and commands
LOAD_WEIGHTS = False  # To load specific weights to the model

path_model = "/models/v15/driverless_model_v15_2.h5"
path_weight = "/weights/v9/weights-improvement-v9_3-13-2.61.hdf5"

layer_name = "cropping2d_1"


class cmd_vel_node(object):
    def __init__(self):

        # Setup Paths
        rospack = rospkg.RosPack()
        path_imitation_learning = rospack.get_path('imitation_learning')
        self.model_path = path_imitation_learning + path_model  # Paths to modify!!
        self.model_weights_path = path_imitation_learning + path_weight
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
        self.image_sub = rospy.Subscriber("/fs/c1/image_raw", Image, self.cvt_image)

        self.im_pre = image_preprocessing.process()

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
        if LOAD_WEIGHTS is True:
            model.load_weights(self.model_weights_path)
        print("Model loaded.")

        model.summary()

        # Set up a model that returns the activation values for our target layer
        layer = model.get_layer(name=layer_name)
        feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

        while not rospy.is_shutdown():
            # Only run loop if we have an image
            if self.imgRcvd:

                # preprocessed_image, preprocessed_image_scaled = self.im_pre.process_img(self.latest_image, pixelation,
                #                                                                        screen_size)
                preprocessed_image = np.expand_dims(self.latest_image, axis=0)  # we add batch dimension
                # preprocessed_image = np.expand_dims(preprocessed_image, axis=3)  # we add color dimension (if grayscale)

                # Prediction of Neural Network based on image captured
                # pred[0]=angle,  pred[1]=speed
                pred = model.predict(preprocessed_image)

                # Publish
                self.cmdvel.angular.z = pred[0][0]
                self.cmdvel.linear.x = pred[0][1]
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

# class cmd_vel_node(object):
#     def __init__(self):
#
#         """ROS Subscriptions """
#         # self.joy_sub = rospy.Subscriber("/joy_teleop/cmd_vel_stamped",TwistStamped,self.debug_img)
#         self.cmd_sub = rospy.Subscriber("/fs/cmd_vel", Twist, self.debug_img)
#         self.debug_pub = rospy.Publisher("/image_converter/debug_video", Image, queue_size=10)
#
#         self.image_sub = rospy.Subscriber("/fs/c1/image_raw", Image, self.cvt_image)
#         self.image_pub = rospy.Publisher("/image_converter/output_video", Image, queue_size=10)
#         self.cmdVel_pub = rospy.Publisher("/fs/cmd_vel", Twist, queue_size=10)
#         self.cmdVelStamped_pub = rospy.Publisher('/fs/cmd_vel_stamped', TwistStamped, queue_size=10)
#
#         """ Variables """
#         self.model_path = '/home/agus/ros/fss_ws/src/imitation_learning/scripts/driverless_model.h5'
#         self.cmdvel = Twist()
#         self.baseVelocity = TwistStamped()
#         self.input_cmd = TwistStamped()
#         self.bridge = CvBridge()
#         self.latestImage = None
#         self.outputImage = None
#         self.resized_image = None
#         self.debugImage = None
#         self.imgRcvd = False
#
#     def debug_img(self, cmd):
#         self.input_cmd = cmd
#         throttle = self.input_cmd.linear.x
#         steering = self.input_cmd.angular.z
#
#         # print("CMD: {} {}").format(throttle,steering)
#
#         if self.imgRcvd:
#             # Get latest image
#             # self.debugImage = cv2.resize(self.latestImage, (320,180))
#             self.debugImage = self.latestImage
#             height, width, channels = self.debugImage.shape
#
#             # Text settings
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             location = (50, 50)  # 10,20
#             fontScale = .5
#             fontColor = (255, 0, 0)
#             lineType = 2
#             throttle_str = "Throttle: " + "{0:.2f}".format(throttle)
#             steering_str = "Steering: " + "{0:.2f}".format(steering)
#
#             # Print text
#             cv2.putText(self.debugImage, throttle_str, location, font, fontScale, fontColor, lineType)
#             cv2.putText(self.debugImage, steering_str, (10, 35), font, fontScale, fontColor, lineType)
#
#             # Draw markers
#             throttle_center = int(50 + (120 - (120 * (throttle / .15))))
#
#             radius = 3
#             circleColor = (0, 0, 255)
#             thickness = -1
#
#             # cv2.circle(self.debugImage, (20, throttle_center), radius, circleColor, thickness, lineType, shift=0)
#
#             steering_center = int(160 + (140 * (steering / 1.6)))
#
#             # cv2.circle(self.debugImage, (steering_center, 160), radius, circleColor, thickness, lineType, shift=0)
#
#             # Publish debug image
#             self.publish(self.debugImage, self.bridge, self.debug_pub)
#
#     def cvt_image(self, data):
#         try:
#             self.latestImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
#         except CvBridgeError as e:
#             print(e)
#         if self.imgRcvd != True:
#             self.imgRcvd = True
#
#     def publish(self, image, bridge, publisher):
#         try:
#             # Determine Encoding
#             if np.size(image.shape) == 3:
#                 imgmsg = bridge.cv2_to_imgmsg(image, "bgr8")
#             else:
#                 imgmsg = bridge.cv2_to_imgmsg(image, "mono8")
#             publisher.publish(imgmsg)
#         except CvBridgeError as e:
#             print(e)
#
#     def cmdVel_publish(self, cmdVelocity):
#
#         # Publish Twist
#         self.cmdVel_pub.publish(cmdVelocity)
#
#         # Publish TwistStamped
#         self.baseVelocity.twist = cmdVelocity
#
#         baseVelocity = TwistStamped()
#
#         baseVelocity.twist = cmdVelocity
#
#         now = rospy.get_rostime()
#         baseVelocity.header.stamp.secs = now.secs
#         baseVelocity.header.stamp.nsecs = now.nsecs
#         self.cmdVelStamped_pub.publish(baseVelocity)
#
#     def run(self):
#
#         # check that model Keras version is same as local Keras version
#         f = h5py.File('/home/agus/ros/fss_ws/src/imitation_learning/scripts/driverless_model.h5', mode='r')
#         model_version = f.attrs.get('keras_version')
#         keras_version_installed = None
#         keras_version_installed = str(keras_version).encode('utf8')
#
#         if model_version != keras_version_installed:
#             print(
#             'You are using Keras version ', keras_version_installed, ', but the model was built using ', model_version)
#
#         # Model reconstruction from JSON file
#
#         with open('/home/agus/ros/fss_ws/src/imitation_learning/scripts/driverless_model.json', 'r') as f:
#             model = model_from_json(f.read())
#
#         model = load_model('/home/agus/ros/fss_ws/src/imitation_learning/scripts/driverless_model.h5')
#
#         # Load weights into the new model
#         print("Model loaded.")
#
#         while True:
#             # Only run loop if we have an image
#             if self.imgRcvd:
#                 # step 1:
#                 # self.resized_image = cv2.resize(self.latestImage, (320,180))
#
#                 self.resized_image = self.latestImage
#
#                 # step 2:
#                 image_array = np.asarray(self.resized_image)
#
#                 # step 3:
#
#                 self.cmdvel.linear.x = 0.5
#                 self.angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
#                 self.angle = -1 if self.angle < -1 else 1 if self.angle > 1 else self.angle
#                 self.cmdvel.angular.z = self.angle
#
#                 # print(self.cmdvel.angular.z)
#
#                 self.cmdVel_publish(self.cmdvel)
#
#                 # Publish Processed Image
#                 self.outputImage = self.latestImage
#                 self.publish(self.outputImage, self.bridge, self.image_pub)
#
#
# def main(args):
#     rospy.init_node('model_control_node', anonymous=True)
#
#     cmd = cmd_vel_node()
#
#     cmd.run()
#
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main(sys.argv)
