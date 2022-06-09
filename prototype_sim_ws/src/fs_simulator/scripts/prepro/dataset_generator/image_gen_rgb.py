import roslaunch
import rospy
import rosservice
from pose_manager import get_pose, get_num_poses
import subprocess
import signal
import os
import glob
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import rospkg

dataset_folder = 'training'
sub_folder = 'Target_RGB'

rospack = rospkg.RosPack()
PATH_fs_simulator = rospack.get_path('fs_simulator')

bridge = CvBridge()
rospy.sleep(25)
poses_file = open(PATH_fs_simulator + '/scripts/prepro/dataset_generator/poses.txt', 'r')
num_poses = get_num_poses(poses_file)

if not os.path.exists(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set'):
    os.mkdir(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set')
if not os.path.exists(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set/' + sub_folder):
    os.mkdir(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set/' + sub_folder)

child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro_simplified.launch"], stdout=subprocess.PIPE)

print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)

rospy.sleep(20)  # 10

# We iterate over each selected pose
for pose_num in range(num_poses):
    position, orientation = get_pose(poses_file, pose_num)
    rosservice.call_service('/gazebo/set_model_state', [
        ['robot', [position, orientation], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']])
    rospy.sleep(0.3)
    data = rospy.wait_for_message('/fs/c1/image_raw', Image)
    cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)
    # cv2.imshow('image', cv_image)
    # cv2.waitKey(2)
    # Get file number
    # list_files = os.listdir('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set/Noise/mid' + str(pose_num))

    if not os.path.exists(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set/' + sub_folder + '/mid' + str(pose_num)):
        os.mkdir(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set/' + sub_folder + '/mid' + str(pose_num))

    list_files = glob.glob(PATH_fs_simulator + '/scripts/prepro/dataset_images/' + dataset_folder + '_set/Noise/mid' + str(pose_num) + '/*')
    # if len(list_files) == 0:
    #     texture_iter = 0
    # else:
    #     val = max(list_files, key=os.path.getctime)
    #     if dataset_folder == 'training':
    #         texture_iter = int(val[115+2*(len(str(pose_num))-1):len(val)-4])
    #     elif dataset_folder == 'test':
    #         texture_iter = int(val[111 + 2 * (len(str(pose_num)) - 1):len(val) - 4])
    texture_iter = len(list_files) - 1
    cv2.imwrite('../dataset_images/' + dataset_folder + '_set/' + sub_folder + '/mid' + str(pose_num) + '/mid'+str(pose_num) + '_' + str(texture_iter) + '.jpg', cv_image)

child.terminate()

