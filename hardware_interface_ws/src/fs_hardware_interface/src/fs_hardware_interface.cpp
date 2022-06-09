#include <sstream>
#include <fs_hardware_interface/fs_hardware_interface.h>

#include <fscpp/fs.h>
#include <fscpp/joint.h>

using namespace hardware_interface;
using joint_limits_interface::JointLimits;
using joint_limits_interface::SoftJointLimits;
using joint_limits_interface::PositionJointSoftLimitsHandle;
using joint_limits_interface::PositionJointSoftLimitsInterface;
using joint_limits_interface::VelocityJointSaturationHandle;
using joint_limits_interface::PositionJointSaturationHandle;

namespace fs_hardware_interface
{
    FSHardwareInterface::FSHardwareInterface(ros::NodeHandle& nh) : nh_(nh)
    {
        init();
        controller_manager_.reset(new controller_manager::ControllerManager(this, nh_));
        nh_.param("/fs/hardware_interface/loop_hz", loop_hz_, 0.1);
        ROS_DEBUG_STREAM_NAMED("constructor","Using loop frequency of " << loop_hz_ << " hz");
        ros::Duration update_freq = ros::Duration(1.0/loop_hz_);
        non_realtime_loop_ = nh_.createTimer(update_freq, &FSHardwareInterface::update, this);
    }

    FSHardwareInterface::~FSHardwareInterface()
    {

    }

    void FSHardwareInterface::init()
    {
        // Get joint names
        nh_.getParam("/fs/hardware_interface/joints", joint_names_);
        if (joint_names_.empty())
        {
            ROS_FATAL_STREAM_NAMED("init","No joints found on parameter server for controller. Did you load the proper yaml file?");
        }
        num_joints_ = joint_names_.size();

        // Resize vectors
        joint_position_.resize(num_joints_);
        joint_velocity_.resize(num_joints_);
        joint_effort_.resize(num_joints_);
        joint_position_command_.resize(num_joints_);
        joint_velocity_command_.resize(num_joints_);
        joint_effort_command_.resize(num_joints_);

        // Initialize Controller
        for (int i = 0; i < num_joints_; ++i)
        {
            fscpp::Joint joint = fs.getJoint(joint_names_[i]);


            ROS_DEBUG_STREAM_NAMED("constructor","Loading joint name: " << joint.name);



            nh_.getParam("/fs/joint_offsets/" + joint.name, joint.angleOffset);
            nh_.getParam("/fs/joint_read_ratio/" + joint.name, joint.readRatio);
            nh_.getParam("/fs/joint_actuator_type/" + joint.name, joint.actuatorType);

            joint.setActuatorType(joint.actuatorType);
            fs.setJoint(joint);

            // Create joint state interface
            JointStateHandle jointStateHandle(joint.name, &joint_position_[i], &joint_velocity_[i], &joint_effort_[i]);
            joint_state_interface_.registerHandle(jointStateHandle);

            switch (joint.getActuatorType()) {
                case ACTUATOR_TYPE_VELOCITY_MOTOR: {

                    // Create effort joint interface
                    JointHandle jointEffortHandle(jointStateHandle, &joint_effort_command_[i]);
                    effort_joint_interface_.registerHandle(jointEffortHandle);

                    break;
                }

                case ACTUATOR_TYPE_POSITION_MOTOR: {

                    // Create effort joint interface
                    JointHandle jointEffortHandle(jointStateHandle, &joint_effort_command_[i]);
                    effort_joint_interface_.registerHandle(jointEffortHandle);

                    break;
                }

                case ACTUATOR_TYPE_VELOCITY_SERVO: {

                    // Create velocity joint interface
                    JointHandle jointVelocityHandle(jointStateHandle, &joint_velocity_command_[i]);
                    velocity_joint_interface_.registerHandle(jointVelocityHandle);

                    // Create Joint Limit interface
                    JointLimits limits;
                    getJointLimits(joint.name, nh_, limits);
                    VelocityJointSaturationHandle jointLimitsHandle(jointVelocityHandle, limits);
                    velocity_joint_saturation_interface_.registerHandle(jointLimitsHandle);


                    break;
                }


                case ACTUATOR_TYPE_POSITION_SERVO: {

                    // Create position joint interface
                    JointHandle jointPositionHandle(jointStateHandle, &joint_position_command_[i]);
                    position_joint_interface_.registerHandle(jointPositionHandle);

                    // Create Joint Limit interface
                    JointLimits limits;
                    SoftJointLimits softLimits;
                    if (!getJointLimits(joint.name, nh_, limits)) {
                        ROS_ERROR_STREAM("Cannot set joint limits for " << joint.name);
                    } else {
                        PositionJointSaturationHandle jointLimitsHandle(jointPositionHandle, limits);
                        position_joint_saturation_interface_.registerHandle(jointLimitsHandle);
                    }

                    break;
                }


            }



        }

        registerInterface(&joint_state_interface_);
        registerInterface(&position_joint_interface_);
        registerInterface(&velocity_joint_interface_);
        registerInterface(&effort_joint_interface_);
        registerInterface(&position_joint_soft_limits_interface_);
        registerInterface(&velocity_joint_saturation_interface_);
    }

    void FSHardwareInterface::update(const ros::TimerEvent& e)
    {
        _logInfo = "\n";
        _logInfo += "Joint Position Command:\n";
        for (int i = 0; i < num_joints_; i++)
        {
            std::ostringstream jointPositionStr;
            jointPositionStr << joint_position_command_[i];
            _logInfo += "  " + joint_names_[i] + ": " + jointPositionStr.str() + "\n";
        }

        elapsed_time_ = ros::Duration(e.current_real - e.last_real);

        read();
        controller_manager_->update(ros::Time::now(), elapsed_time_);
        write(elapsed_time_);

        //ROS_INFO_STREAM(_logInfo);
    }

    void FSHardwareInterface::read()
    {
        _logInfo += "Joint State:\n";
        for (int i = 0; i < num_joints_; i++)
        {
            fscpp::Joint joint = fs.getJoint(joint_names_[i]);

            //Faltaria leer velocidad del encoder
            ROS_INFO("Trying to read MOTOR_ID=%i", joint.getMotorId());
            joint_position_[i] = joint.readAngle();
            std::ostringstream jointPositionStr;
            jointPositionStr << joint_position_[i];
            _logInfo += "  " + joint.name + ": " + jointPositionStr.str() + "\n";

            fs.setJoint(joint);

        }
    }

    void FSHardwareInterface::write(ros::Duration elapsed_time)
    {
        position_joint_saturation_interface_.enforceLimits(elapsed_time);
        velocity_joint_saturation_interface_.enforceLimits(elapsed_time);


        for (int i = 0; i < num_joints_; i++)
        {
            fscpp::Joint joint = fs.getJoint(joint_names_[i]);
            //if (joint_effort_command_[i] > 1) joint_effort_command_[i] = 1;
            //if (joint_effort_command_[i] < -1) joint_effort_command_[i] = -1;


            switch (joint.getActuatorType()) {
                case ACTUATOR_TYPE_VELOCITY_MOTOR: {

                    double effort = joint_effort_command_[i];
                    uint8_t duration = 15;
                    double previousEffort = joint.getPreviousEffort();
                    effort += previousEffort;
                    joint.actuate(effort, duration);
                    std::ostringstream jointEffortStr;
                    jointEffortStr << joint_effort_command_[i];
                    _logInfo += "  " + joint.name + ": " + jointEffortStr.str() + "\n";


                    break;
                }

                case ACTUATOR_TYPE_POSITION_MOTOR: {

                    double effort = joint_effort_command_[i];
                    uint8_t duration = 15;
                    double previousEffort = joint.getPreviousEffort();
                    effort += previousEffort;
                    joint.actuate(effort, duration);
                    std::ostringstream jointEffortStr;
                    jointEffortStr << joint_effort_command_[i];
                    _logInfo += "  " + joint.name + ": " + jointEffortStr.str() + "\n";


                    break;
                }

                case ACTUATOR_TYPE_VELOCITY_SERVO: {

                    double velocity = joint_velocity_command_[i];
                    ROS_DEBUG("Velocity Command Joint Velocity: %f  Motor ID: %i \n", velocity, joint.getMotorId());
                    uint8_t duration = 15;
                    auto command_ang=(int16_t)angles::to_degrees(velocity);
                    ROS_DEBUG("Velocity Command_ANG Joint Velocity: %f  Motor ID: %i \n", velocity, joint.getMotorId());
                    joint.actuate(command_ang, duration);
                    std::ostringstream jointEffortStr;
                    jointEffortStr << joint_effort_command_[i];
                    _logInfo += "  " + joint.name + ": " + jointEffortStr.str() + "\n";



                    break;
                }


                case ACTUATOR_TYPE_POSITION_SERVO: {

                    double position = joint_position_command_[i];
                    ROS_DEBUG("Position Command Joint Steer: %f  Motor ID: %i \n", position, joint.getMotorId());
                    uint8_t duration = 15;
                    auto command_ang=(int16_t)angles::to_degrees(position);
                    joint.actuate(command_ang, duration);
                    std::ostringstream jointEffortStr;
                    jointEffortStr << joint_effort_command_[i];
                    _logInfo += "  " + joint.name + ": " + jointEffortStr.str() + "\n";


                    break;
                }
            }


        }
    }
}