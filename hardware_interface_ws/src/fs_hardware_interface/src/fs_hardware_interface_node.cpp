#include <fs_hardware_interface/fs_hardware_interface.h>
int main(int argc, char** argv)
{
    ros::init(argc, argv, "fs_hardware_interface");
    ros::NodeHandle nh;

    // NOTE: We run the ROS loop in a separate thread as external calls such
    // as service callbacks to load controllers can block the (main) control loop

    //ros::AsyncSpinner spinner(1);

    ros::MultiThreadedSpinner spinner(2); // Multiple threads for controller service callback and for the Service client callback used to get the feedback from ardiuno

    //spinner.start();

    fs_hardware_interface::FSHardwareInterface fs(nh);

    spinner.spin();

    //ros::spin();

    return 0;
}