import roslaunch
import rospy
import rosservice
from pose_manager import get_pose, get_num_poses


def get_images_over_texture():
    poses_file = open('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/poses.txt', 'r')
    num_poses = get_num_poses(poses_file)

    # rospy.init_node('Image_GAN', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/launch/prepro.launch"])
    launch.start()
    rospy.loginfo("started")

    rospy.sleep(8)  # 10

    # We iterate over each selected pose
    # rosservice.call_service('/gazebo/set_model_state', [['robot', [[31.0, 0.06, 0.05], [0.0, 0.0, 0.36, 0.93]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']])
    for pose_num in range(num_poses):
        position, orientation = get_pose(poses_file, pose_num)
        rosservice.call_service('/gazebo/set_model_state', [
            ['robot', [position, orientation], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']])
        rospy.sleep(1)

    # # 3 seconds later
    launch.shutdown()
    # rospy.sleep(10)


# Esto no funca
# rospy.sleep(30)
# # # 3 seconds later
# launch.start()
#
# rospy.sleep(10)
# # # 3 seconds later
# launch.shutdown()