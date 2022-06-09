import roslaunch
import rospy
import rosservice
from pose_manager import get_pose, get_num_poses
import subprocess
import signal
import os



# roscore = subprocess.Popen('roscore', stdout=subprocess.PIPE)
rospy.sleep(15)
poses_file = open('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/poses.txt', 'r')
num_poses = get_num_poses(poses_file)

# child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro.launch"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
# child = subprocess.Popen("roslaunch fs_simulator prepro.launch", stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro.launch"], stdout=subprocess.PIPE)
# child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro.launch"], stdout=subprocess.PIPE)  #  , preexec_fn=os.setsid)

print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)

rospy.sleep(10)  # 10

# We iterate over each selected pose
rosservice.call_service('/gazebo/set_model_state', [['robot', [[31.0, 0.06, 0.05], [0.0, 0.0, 0.36, 0.93]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']])
for pose_num in range(num_poses):
    position, orientation = get_pose(poses_file, pose_num)
    rosservice.call_service('/gazebo/set_model_state', [
        ['robot', [position, orientation], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']])
    rospy.sleep(0.3)

# # 3 seconds later
# child.send_signal(signal.SIGINT)
# os.system("rosnode kill -a")
#You may also use .terminate() method
child.terminate()
# rospy.sleep(10)
# child.kill()
# child.communicate()
# os.killpg(os.getpgid(child.pid), signal.SIGTERM)
# roscore.terminate()
# Kill process.
