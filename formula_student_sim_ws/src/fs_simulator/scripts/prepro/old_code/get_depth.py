import roslaunch
import rospy
import rosservice
from pose_manager import get_pose, get_num_poses
import subprocess
import signal
import os
import glob
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt


bridge = CvBridge()

# rospy.sleep(15)
poses_file = open('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/poses.txt', 'r')
num_poses = get_num_poses(poses_file)

child = subprocess.Popen(["roslaunch", "fs_simulator", "prepro.launch"], stdout=subprocess.PIPE)

print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)

rospy.sleep(10)  # 10

if not os.path.exists('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set'):
    os.mkdir('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set')
if not os.path.exists('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set/Target_Depth'):
    os.mkdir('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set/Target_Depth')

# We iterate over each selected pose
for pose_num in range(num_poses):
    position, orientation = get_pose(poses_file, pose_num)
    rosservice.call_service('/gazebo/set_model_state', [
        ['robot', [position, orientation], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']])
    rospy.sleep(0.3)
    data = rospy.wait_for_message('/camera/depth/image_raw', Image)
    cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
    # cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY, cv_image)
    # cv2.imshow('image', cv_image)
    # cv2.waitKey(2)
    # Get file number
    # list_files = glob.glob('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set/Noise/mid' + str(pose_num) + '/*')
    # val = max(list_files, key=os.path.getctime)
    # texture_iter = int(val[115+2*(len(str(pose_num))-1):len(val)-4])+1

    depth_array = np.array(cv_image, dtype=np.float32)  # Este tiene la distancia en metros a cada pixel.
    depth_array[np.isnan(depth_array)] = 0.0
    # Guardamos numpy array con prcision de punto flotante
    depth_array_32fc1 = depth_array.copy()
    # Aca antes de normalizar se podria sacar el valor maximo (que luego va a corresponder con el 255). De esta manera se puede despues estimar (sin mucha precision) la distancia de conos.
    cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
    # Al guardar la imagen se comprime a 8 bits por pixel. Se podria analizar mantener buena resolucion.

    if not os.path.exists('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set/Target_Depth/mid' + str(pose_num)):
        os.mkdir('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/training_set/Target_Depth/mid' + str(pose_num))

    # Guardamos el point cloud
    np.save('dataset_images/training_set/Target_Depth/mid' + str(pose_num) + '/mid'+str(pose_num) + '_' + str(0), depth_array_32fc1)
    # Guardamos imagen de referencia. (No deberia ser utilziada para entrenar, dado que tiene precision de 8 bits)
    cv2.imwrite('dataset_images/training_set/Target_Depth/mid' + str(pose_num) + '/mid'+str(pose_num) + '_reference_' + str(0) + '.jpg', depth_array*255)

child.terminate()

