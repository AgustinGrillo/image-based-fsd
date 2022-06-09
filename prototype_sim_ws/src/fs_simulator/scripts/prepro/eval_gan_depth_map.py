#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
import math

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import rospy
import rospkg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, RegionOfInterest
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError


tf.compat.v1.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)


# Load model
rospack = rospkg.RosPack()
package_path = rospack.get_path('fs_simulator')
generator_model = tf.keras.models.load_model(package_path + '/scripts/prepro/gan_models/model_test_320x640_blank_dataset_V1/generator_model_epoch_30', compile=False)

# Load image
# test_img_path = 'dataset_images/data_set_4/training_set/Noise/mid8/mid8_0.jpg'
test_img_path = 'dataset_images/real_test_images/prototype/0005.jpg'
test_video_path = 'dataset_images/real_test_images/formula/AMZ_driverless.mp4'


class DepthEstimator(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(test_video_path)
        self.pub_simple_img = rospy.Publisher("image_simplified", Image, queue_size=10)
        self.pub_depth_img = rospy.Publisher("image_rect", Image, queue_size=10)
        self.pub_info = rospy.Publisher("camera_info", CameraInfo, queue_size=10)
        self.bridge = CvBridge()

        self.counter = 0

    def estimate(self):
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            if self.cap.isOpened():
                ret, frame = self.cap.read()

                if frame is None:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(test_video_path)
                    ret, frame = self.cap.read()
                img = frame[250:442, :640, :]

                # DEBUG
                frame = cv2.imread(test_img_path)
                img = frame[20:340, :, :]
                # DEBUG

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_3d = img.astype(np.float32)
                img_3d = img_3d / 255.0
                # pts_1 = [[146,155],  [115,192], [390,192], [264, 94]]
                # pts_2 = [[0, 0], [0, 40], [320, 0]]
                # pts_3 = [[640, 0], [320, 0], [640, 40]]
                # img_3d = cv2.fillPoly(img_3d, np.array([pts_1]), (0.5, 0.5, 0.5))
                # img_3d = cv2.fillPoly(img_3d, np.array([pts_2]), (0.5, 0.5, 0.5))
                # img_3d = cv2.fillPoly(img_3d, np.array([pts_3]), (0.5, 0.5, 0.5))
                img_3d = 2 * img_3d - 1.0
                img = np.expand_dims(img_3d, 0)

                gen_img = generator_model(img, training=False)
                sim = gen_img[0][0, :, :, :]
                depth = gen_img[1][0, :, :, 0]

                sim_np = sim.numpy()
                depth_np = depth.numpy()
                input_img = img_3d


                sim_np = 0.5 * (sim_np + 1.0) * 255
                depth_np = 0.5 * (depth_np + 1.0) * 4.0

                sim_extender_mask = np.empty((360, 640, 3))
                depth_extender_mask = np.empty((360, 640))
                sim_extender_mask[:] = 0.0
                depth_extender_mask[:] = 0.0
                sim_extender_mask[20:340, :, :] = sim_np
                depth_extender_mask[20:340, :] = depth_np
                sim_np = copy.copy(sim_extender_mask)
                depth_np = copy.copy(depth_extender_mask)

                #DEBUG
                # sim_np = cv2.imread('dataset_images/data_set_2/training_set/Target_Simplified/mid0/mid0_0.jpg')
                # depth_np = np.load('dataset_images/data_set_2/training_set/Target_Depth/mid0/mid0_0.npy')
                #DEBUG

                sim_np = cv2.cvtColor(sim_np.astype(dtype=np.uint8), cv2.COLOR_RGB2BGR)

                sim = self.bridge.cv2_to_imgmsg(sim_np, "bgr8")
                sim.header.frame_id = "/camera3d_link"
                sim.header.stamp = rospy.Time.from_sec(time.time())
                sim.header.seq =self.counter

                depth = self.bridge.cv2_to_imgmsg(depth_np.astype(dtype=np.float32), "passthrough") # "passthrough"  "32FC1"
                depth.header.frame_id = "/camera3d_link"
                depth.header.stamp = rospy.Time.from_sec(time.time())
                depth.header.seq = self.counter

                self.pub_simple_img.publish(sim)
                self.pub_depth_img.publish(depth)



            # Camera info
            camera_info_message = CameraInfo()
            camera_info_header = Header()

            # camera_info_header.seq = self.counter
            camera_info_header.stamp = rospy.Time.from_sec(time.time())
            camera_info_header.frame_id = "/camera3d_link"
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
            self.pub_info.publish(camera_info_message)

            self.counter += 1

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('depth_evaluator', anonymous=True)
    depth_estimator = DepthEstimator()
    try:
        depth_estimator.estimate()
    except rospy.ROSInterruptException:
        pass
