#!/usr/bin/env python

# Very simple node to publish a path to be followed

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

def talker():
    pub = rospy.Publisher('/fs/path', PoseArray, queue_size=10)
    rospy.init_node('Path_Publisher', anonymous=True)

    path = PoseArray()
    p = Pose()

    p.position.x = 0
    p.position.y = 0
    path.poses.append(p)

    p = Pose()
    p.position.x = 20
    p.position.y = 0
    path.poses.append(p)

    p = Pose()
    p.position.x = 30
    p.position.y = 10
    path.poses.append(p)

    p = Pose()
    p.position.x = 30
    p.position.y = 30
    path.poses.append(p)

    p = Pose()
    p.position.x = 20
    p.position.y = 40
    path.poses.append(p)

    p = Pose()
    p.position.x = 0
    p.position.y = 40
    path.poses.append(p)

    p = Pose()
    p.position.x = -10
    p.position.y = 30
    path.poses.append(p)

    p = Pose()
    p.position.x = -10
    p.position.y = 10
    path.poses.append(p)

    p = Pose()
    p.position.x = 0
    p.position.y = 0
    path.poses.append(p)

    if not rospy.is_shutdown():
        rospy.sleep(2) #new subscribesrs take some time to be noticed by rosmaster, up to 5 sec.
        pub.publish(path)
        rospy.loginfo("Path Published")

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
