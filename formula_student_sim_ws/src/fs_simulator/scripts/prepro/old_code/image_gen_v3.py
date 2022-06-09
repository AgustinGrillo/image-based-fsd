#!/usr/bin/env python

import rospy
import subprocess
import signal

child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro.launch"])
# child.wait() #You can use this line to block the parent process untill the child process finished.
print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)

rospy.sleep(8)

child.send_signal(signal.SIGINT) #You may also use .terminate() method
#child.terminate()

child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro.launch"])
# child.wait() #You can use this line to block the parent process untill the child process finished.
print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)

rospy.sleep(8)

child.send_signal(signal.SIGINT) #You may also use .terminate() method

#for more: https://docs.python.org/2/library/subprocess.html