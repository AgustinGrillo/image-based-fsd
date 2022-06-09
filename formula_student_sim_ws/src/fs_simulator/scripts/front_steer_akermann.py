#!/usr/bin/env python
import rospy
import math

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

import tf
from geometry_msgs.msg import Transform

pubR = rospy.Publisher('/fs/right_front_steer/command', Float64, queue_size=1)
pubL = rospy.Publisher('/fs/left_front_steer/command', Float64, queue_size=1)
pubM = rospy.Publisher('/fs/motor_controller/cmd_vel', Twist, queue_size=1)

# These should always be modified in parallel with the ones in urdf/urdf.xacro file
g = 1.729  # wheel_base
d = 1.186 / 2  # axle_track


def callback(data):
    angle = data.angular.z
    linVel = data.linear.x

    if abs(angle) > 0.001:

        if angle <= 0:

            if angle < -1:
                aL = -math.pi / 4
            else:
                aL = math.pi / 4 * angle

            Rad = g / math.tan(aL) - d
            aR = math.atan(g / (Rad - d))

        else:

            if angle > 1:
                aR = math.pi / 4
            else:
                aR = math.pi / 4 * angle

            Rad = g / math.tan(aR) + d
            aL = math.atan(g / (Rad + d))

        angle_vel = linVel / Rad

    else:
        aR = 0
        aL = 0
        angle_vel = 0

        # rospy.loginfo('g'+str(g)+'d'+str(d)+'angvel'+str(angVel)+'linvel'+str(linVel)+'ar'+str(aR))

    if abs(aR) > math.pi / 4 or abs(aL) > math.pi / 4:
        rospy.loginfo("wheel angle limit exceeded, clamping at maximum")
        # rospy.loginfo('g:'+str(g)+' d:'+str(d)+' al:'+str(aL)+' ar:'+str(aR))

    mes = Twist()
    mes.angular.z = angle_vel
    mes.linear.x = linVel

    pubM.publish(mes)

    pubR.publish(aR)
    pubL.publish(aL)

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

"""
import rospy
import math

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

import tf
from geometry_msgs.msg import Transform

pubR = rospy.Publisher('/fs/right_front_steer/command', Float64, queue_size=1)
pubL = rospy.Publisher('/fs/left_front_steer/command', Float64, queue_size=1)


#These should always be modified in parallel with the ones in urdf/urdf.xacro file
g = 1.729 # wheel_base
d = 1.186/2 # axle_track


def callback(data):

    angVel = data.angular.z
    linVel = data.linear.x



    if abs(linVel) < 0.001 and  abs(angVel) > 0.001:
        aR = math.copysign(math.atan2(g,abs(1/angVel)+math.copysign(d,angVel)),angVel)
        aL = math.copysign(math.atan2(g,abs(1/angVel)-math.copysign(d,angVel)),angVel)
        rospy.loginfo("front wheel limits exceeded, linear vel = 0 ")

    elif abs(angVel) < 0.001:
        aR = 0
        aL = 0

    else:
        aR = math.copysign(math.atan2(g,abs(linVel/angVel)-math.copysign(d,angVel*linVel)),angVel*linVel)
        aL = math.copysign(math.atan2(g,abs(linVel/angVel)+math.copysign(d,angVel*linVel)),angVel*linVel)
        #rospy.loginfo('g'+str(g)+'d'+str(d)+'angvel'+str(angVel)+'linvel'+str(linVel)+'ar'+str(aR))

    if abs(aR) > math.pi/4 or abs(aL) > math.pi/4:
        rospy.loginfo("wheel angle limit exceeded, clamping at maximum")
        #rospy.loginfo('g:'+str(g)+' d:'+str(d)+' al:'+str(aL)+' ar:'+str(aR))

    #rospy.loginfo('g'+str(g)+'d'+str(d)+'angvel'+str(angVel)+'linvel'+str(linVel)+'ar'+str(aR))
    pubR.publish(aR)
    pubL.publish(aL)





def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('front_steer_akermann', anonymous=True)


    rospy.Subscriber("/fs/motor_controller/cmd_vel", Twist, callback)


    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
"""
