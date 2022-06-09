#ifndef ROS_CONTROL__FS_HARDWARE_INTERFACE_H
#define ROS_CONTROL__FS_HARDWARE_INTERFACE_H


#include <fs_hardware_interface/fs_hardware.h>

using namespace hardware_interface;
using joint_limits_interface::JointLimits;
using joint_limits_interface::SoftJointLimits;
using joint_limits_interface::PositionJointSoftLimitsHandle;
using joint_limits_interface::PositionJointSoftLimitsInterface;


namespace fs_hardware_interface
{
    static const double POSITION_STEP_FACTOR = 10;
    static const double VELOCITY_STEP_FACTOR = 10;

    class FSHardwareInterface: public fs_hardware_interface::FSHardware
    {
        public:
            FSHardwareInterface(ros::NodeHandle& nh);
            ~FSHardwareInterface();
            void init();
            void update(const ros::TimerEvent& e);
            void read();
            void write(ros::Duration elapsed_time);

        protected:
            fscpp::FS fs;


            ros::NodeHandle nh_;
            ros::Timer non_realtime_loop_;
            ros::Duration control_period_;
            ros::Duration elapsed_time_;


            double loop_hz_;
            boost::shared_ptr<controller_manager::ControllerManager> controller_manager_;
            double p_error_, v_error_, e_error_;
            std::string _logInfo;
    };

}

#endif