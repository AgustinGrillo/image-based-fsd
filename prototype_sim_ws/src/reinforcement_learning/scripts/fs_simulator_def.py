import rospy
import numpy
import time
import math
from gym import spaces
import fs_simulator_robot_env
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from gazebo_msgs.msg import ContactsState
from std_msgs.msg import Header
from other_scripts.task_commons import LoadYamlFileParamsTest
from other_scripts.openai_ros_common import ROSLauncher
import os
from cv_bridge import CvBridge, CvBridgeError


class FsSimulatorMazeEnv(fs_simulator_robot_env.FsSimulatorEnv):
    def __init__(self):
        """
        This Task Env is designed for having the fssimulator in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        # This parameter HAS to be set up in the MAIN launch of the AI RL script
        ros_ws_abspath = rospy.get_param("/fssimulator/ros_ws_abspath", None)  # fs_simulator_openai_qlearn.yaml
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        #####################################
        # Esto creo que no hace falta!!!! ###
        #####################################
        # ROSLauncher(rospackage_name="reinforcement_learning",
        #            launch_file_name="start_world_maze_loop_brick.launch",  # put_robot_in_world.launch
        #            ros_ws_abspath=ros_ws_abspath)

        #########################################

        # Load Params from the desired fssimulatorYaml file
        LoadYamlFileParamsTest(rospackage_name="reinforcement_learning",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="fs_simulator_def.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(FsSimulatorMazeEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        self.n_actions = rospy.get_param('/fssimulator/n_actions')

        # Bridge for visualizing images
        self.bridge = CvBridge()

        high = numpy.array([1.0, 10.0])
        low = numpy.array([-1.0, -10.0])

        self.action_space = spaces.Box(low, high)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        self.n_observations = rospy.get_param('/fssimulator/n_observations')

        high = numpy.full(self.n_observations, 255)
        low = numpy.full(self.n_observations, 0)

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        self.init_linear_forward_speed = rospy.get_param(
            '/fssimulator/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param(
            '/fssimulator/init_linear_turn_speed')

        # Rewards
        self.forwards_reward = rospy.get_param("/fssimulator/forwards_reward")
        self.turn_reward = rospy.get_param("/fssimulator/turn_reward")
        self.end_episode_points = rospy.get_param(
            "/fssimulator/end_episode_points")

        self.cumulated_steps = 0.0

        self.contact_filtered_pub = rospy.Publisher(
            '/fs/bumper/contact_filtered', Bool, queue_size=1)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=-1)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the fssimulator
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>" + str(action))

        angular_speed = action[0]
        linear_speed = action[1]
        self.last_action = action

        # We tell fssimulator the linear and angular speed to set to execute
        self.move_base(linear_speed,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=0)

        rospy.logdebug("END Set Action ==>" + str(action) +
                       ", NAME=" + str(self.last_action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        fssimulatorEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        camera_scan = self.get_camera_rgb_image_raw()

        cv2_scan = self.bridge.imgmsg_to_cv2(camera_scan, "bgr8")  # convert ros image to opencv image

        contact_scan = self.get_contact_bumper()

        rospy.logdebug("BEFORE DISCRET _episode_done==>" +
                       str(self._episode_done))

        self.status_bumper(contact_scan)

        rospy.logdebug("AFTER DISCRET_episode_done==>" + str(self._episode_done))
        rospy.logdebug("END Get Observation ==>")

        return cv2_scan

    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("FsSimulator has crashed into a cone==>" +
                         str(self._episode_done))
        else:
            rospy.logerr("Fssimulator is Ok ==>")

        return self._episode_done

    def _compute_reward(self, observations, done):

        if not done:
            if abs(self.last_action[1]-1) < 0.9:
                reward = 10 * (1-abs(self.last_action[1]-1))
            else:
                reward = -5 * abs(self.last_action[1]-1)

        else:
            reward = -10

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods

    def status_bumper(self, data):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        if data.states == []:
            "No collision"
            collision = False
        else:
            self._episode_done = True
            collision = True

        self.publish_filtered_contact_scan(collision)

        return collision

    def publish_filtered_contact_scan(self, collision):
        self.contact_filtered_pub.publish(collision)
