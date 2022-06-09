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
import os
# import shutil/fss_sim
import numpy as np
import cv2
import image_preprocessing
from keras import models
import rospkg
import keras
import tensorflow as tf

from std_msgs.msg import Header
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import json
from keras.models import model_from_json, load_model
import h5py
from keras import __version__ as keras_version




DEBUG = False  # To visualize the processed camera input and commands

path_model = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/gan_models/model_test_320x640_blank_dataset_V2/generator_model_epoch_49'
layer_name = "conv_2"

###########################
# GPU
tf.compat.v1.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
# tf.compat.v1.keras.backend.set_session(sess)
keras.backend.set_session(sess)

###########################


# Load model

class cmd_vel_node(object):
    def __init__(self):

        # Setup Paths
        rospack = rospkg.RosPack()
        path_imitation_learning = rospack.get_path('imitation_learning')
        # self.model_path = path_imitation_learning + path_model
        self.model_path = path_model
        print 'Loaded Model Path:', self.model_path

        """ Variables """
        self.cmdvel = Twist()
        self.bridge = CvBridge()
        self.img_segmentation = Image()
        self.img_segmentation.width = 640
        self.img_segmentation.height = 320

        self.latest_image = None
        self.preprocessed_image = None
        self.imgRcvd = False

        self.counter = 0

        """ROS Publications """
        self.cmdVel_pub = rospy.Publisher("/fs/cmd_vel", Twist, queue_size=1)
        self.gan_pub_seg = rospy.Publisher("/image_converter/gan_segmentation", Image, queue_size=1)
        self.gan_pub_depth = rospy.Publisher("/image_converter/gan_depth", Image, queue_size=1)
        self.gan_pub_info = rospy.Publisher("camera_info", CameraInfo, queue_size=1)

        if DEBUG is True:
            self.debug_pub = rospy.Publisher("/image_converter/debug_video", Image, queue_size=10)

        """ROS Subscriptions """
        # self.image_sub = rospy.Subscriber("/csi_cam_0/image_rect_color", Image, self.cvt_image)
        self.image_sub = rospy.Subscriber("/pseye_camera/image_raw", Image, self.cvt_image)

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
        model = tf.keras.models.load_model(self.model_path)
        print("Model loaded")

        # model.summary()

        # Set up a model that returns the activation values for our target layer
        if DEBUG is True:
            layer = model.get_layer(name=layer_name)
            feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

        while not rospy.is_shutdown():
            # Only run loop if we have an image
            if self.imgRcvd:

                preprocessed_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # we add batch dimension

                # Prediction of Neural Network based on image captured
                # pred[0]=angle,  pred[1]=speed
                # pred = model.predict(preprocessed_image)
                preprocessed_image = preprocessed_image.astype(np.float32)
                preprocessed_image = preprocessed_image[:, 20:340, :]
                preprocessed_image = preprocessed_image/255.0
                preprocessed_image = (2.0 * preprocessed_image) - 1.0
                gen_img = model(preprocessed_image, training=False)
                sim = gen_img[0][0, :, :, :]
                depth = gen_img[0][0, :, :, 0]
                sim_np = sim.numpy()
                depth_np = depth.numpy()
                sim_np = 0.5*(sim_np + 1.0) * 255.0
                depth_np = 0.5 * (depth_np + 1.0) * 4.0

                sim_np = sim_np.astype(np.uint8)
                sim_np = cv2.cvtColor(sim_np, cv2.COLOR_RGB2BGR)
                tst = self.bridge.cv2_to_imgmsg(sim_np, encoding="passthrough")
                self.gan_pub_seg.publish(tst)

                depth = self.bridge.cv2_to_imgmsg(depth_np.astype(dtype=np.float32), "passthrough")
                depth.header.frame_id = "map"
                depth.header.stamp = rospy.Time.from_sec(time.time())
                self.gan_pub_depth.publish(depth)

                # Camera info
                camera_info_message = CameraInfo()
                camera_info_header = Header()

                # camera_info_header.seq = self.counter
                camera_info_header.stamp = rospy.Time.from_sec(time.time())
                camera_info_header.frame_id = "map"
                camera_info_message.header = camera_info_header
                camera_info_message.width = 640
                camera_info_message.height = 360
                camera_info_message.distortion_model = "plumb_bob"
                camera_info_message.D = [0.0, 0.0, 0.0, 0.0, 0.0]
                camera_info_message.K = [400.0, 0.0, 320.0, 0.0, 400.0, 167.0, 0.0, 0.0, 1.0]
                camera_info_message.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                camera_info_message.P = [400.0, 0.0, 320.0, -0.0, 0.0, 400.0, 167.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                camera_info_message.binning_x = 0
                camera_info_message.binning_y = 0
                # camera_info_message.roi.x_offset = 0
                # camera_info_message.roi.y_offset = 0
                # camera_info_message.roi.height = 0
                # camera_info_message.roi.width = 0
                # camera_info_message.roi.do_rectify_false = False
                self.gan_pub_info.publish(camera_info_message)

                self.counter += 1


                # Publish
                # self.cmdvel.angular.z = pred[0][0]
                # # TODO: Scale accordingly
                # self.cmdvel.linear.x = pred[0][1] * 0.3
                # # self.cmdvel.linear.x = 2 if self.cmdvel.linear.x > 2 else self.cmdvel.linear.x
                # self.cmdVel_pub.publish(self.cmdvel)
                self.imgRcvd = False

                # print 'Speed:', "%.2f" % self.cmdvel.linear.x, 'Angle:', "%.2f" % self.cmdvel.angular.z

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
