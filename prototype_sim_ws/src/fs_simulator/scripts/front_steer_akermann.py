#!/usr/bin/env python
import rospy
import math
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

import tf
from geometry_msgs.msg import Transform

pubR = rospy.Publisher('/fs/right_front_steer/command', Float64, queue_size=1)
pubL = rospy.Publisher('/fs/left_front_steer/command', Float64, queue_size=1)
pubM = rospy.Publisher('/fs/motor_controller/cmd_vel', Twist, queue_size=1)


WB = 0.170  # wheel_base
TW = 0.23  # axle_track
angle_wheel_max = math.pi / 5.14
FH = 0.022  # Front Hub Arm Rotation
LA = 0.0235  # Length arm Servo


def callback(data):
    
    angle_virtual = data.angular.z  # data.angular.z is between -1 and 1 comes from Joystick (Positive Steer LEFT)
    linVel = data.linear.x

    linVel = np.clip(linVel, -0.3, 0.3)  # TODO: Clip according to angle steer.

    if abs(angle_virtual) > 0.001:

        if angle_virtual <= -1:
            angle_wheel = -angle_wheel_max
        elif -1 < angle_virtual < 1:
            angle_wheel = angle_wheel_max * angle_virtual
        elif angle_virtual >= 1:
            angle_wheel = angle_wheel_max

        Rad = WB / math.tan(angle_wheel)  # Should be WB/math.tan(angle_wheel)

        omega = linVel / Rad

    else:
        angle_wheel = 0
        omega = 0

    if abs(angle_wheel) > angle_wheel_max:
        rospy.loginfo("wheel angle limit exceeded, clamping at maximum")
        # rospy.loginfo('g:'+str(g)+' d:'+str(d)+' al:'+str(aL)+' ar:'+str(aR))

    mes = Twist()
    mes.angular.z = omega
    mes.linear.x = linVel

    pubM.publish(mes)

    pubR.publish(angle_wheel)
    pubL.publish(angle_wheel)

    # rospy.loginfo('angvel:'+str(angle_vel)+' linvel:'+str(linVel)+' ar:'+str(aR)+' aL:'+str(aL))


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('front_steer_akermann', anonymous=True)

    rospy.Subscriber("/fs/cmd_vel", Twist, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
