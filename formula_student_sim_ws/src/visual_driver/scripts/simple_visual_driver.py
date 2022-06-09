#!/usr/bin/env python

##############################################
## Quick-and-dirty visual driver algorithm  ##
## to serve as a starting point for further ##
## autonomous software development          ##
##############################################

# some of this libraries might not be nessesary
from __future__ import print_function
import rospy
import math
import numpy as np
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import tf
from geometry_msgs.msg import Transform

pub = rospy.Publisher('/fs/motor_controller/cmd_vel', Twist, queue_size=1)
bridge = CvBridge()
alpha = 0.033 #smoothing of cones center of mass
beta = 0.0045 #responsiveness of the whole algorithm
yY = yB = xY = xB = 320

def callback(data):

    try:
      input_image = bridge.imgmsg_to_cv2(data, "bgr8") #convert ros image to opencv image
    except CvBridgeError as e:
      print(e)

    cv_image = input_image[245:480, 0:640] #crop out sky and horizon

    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) #change to HSV for a better color segmentation

    maskY = cv2.inRange(hsv_image, (20,150,50), (50, 230, 230)) # Yellow
    maskB = cv2.inRange(hsv_image, (100,150,50), (130, 200, 130)) # Blue

    mask = cv2.bitwise_or(maskY, maskB)
    result = cv2.bitwise_and(cv_image, cv_image, mask=mask) # Segmented image, not used in calculations

    # Persistant variables
    global xB
    global xY
    global yB
    global yY
    global alpha
    global beta

    # compute center of mass for blue & Yellow pixels
    mB = cv2.moments(maskB)
    mY = cv2.moments(maskY)
    if abs(mB["m00"])>0.0001:
        xB = xB* (1- alpha)+ alpha* int(mB["m10"] / mB["m00"])
        yB = yB* (1- alpha)+ alpha* int(mB["m01"] / mB["m00"])
    if abs(mY["m00"])>0.0001:
        xY = xY* (1- alpha)+ alpha* int(mY["m10"] / mY["m00"])
        yY = yY* (1- alpha)+ alpha* int(mY["m01"] / mY["m00"])

    aux = Twist()
    aux.angular.z = beta * ((-xB+160)+(-xY+480)) #if cones go right, go right, if left left
    aux.linear.x = 2 - min([ abs( aux.angular.z ),1]) #slow down in curves
    pub.publish(aux)

    #Visualization for debbuging
    cv2.circle(maskY, (int(xY), int(yY)), 8, (100, 100,100), -1)
    cv2.circle(maskB, (int(xB), int(yB)), 8, (100, 100, 100), -1)
    cv2.line(result, (320, 0), (320, 280), (0, 0, 255))
    cv2.arrowedLine(result, (int(aux.angular.z *100)+320, int(aux.linear.x *100)-50),(320, 70) ,(30, 200, 30))
    #print("angular speed: " + str(aux.angular.z))
    w1 = np.hstack((result, input_image[145:380, 0:640] ))
    w2 = np.hstack((maskB,maskY))
    window = np.vstack((w1,cv2.cvtColor(w2, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Visual Driver Formula Driverless",window)
    cv2.waitKey(3)


def listener():

    rospy.init_node('visual_driver', anonymous=False)
    rospy.Subscriber("/fs/c1/image_raw", Image, callback)
    # spin() simply keeps python from running until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
