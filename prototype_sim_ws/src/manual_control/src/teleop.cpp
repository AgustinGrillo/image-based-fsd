#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Twist.h"

double speed=0;
double angle=0;

void pointCallback(const geometry_msgs::Twist  msg_rec)
{
  //ROS_INFO("I heard %f %f",msg_rec.linear.x,msg_rec.angular.z);

  speed=speed+0.5*msg_rec.linear.x;
  angle=angle+0.05*msg_rec.angular.z;

}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "teleop");

  geometry_msgs::Twist msg;

  msg.linear.x=0;
  msg.linear.y=0;
  msg.linear.z=0;
  msg.angular.x=0;
  msg.angular.y=0;
  msg.angular.z=0;

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("cmd_vel", 10, pointCallback);
  ros::Publisher chatter_pub = n.advertise<geometry_msgs::Twist>("/fs/cmd_vel", 10);

  ros::Rate loop_rate(5);
  while (ros::ok())
  {
    msg.linear.x = speed;
    msg.angular.z = angle;
    chatter_pub.publish(msg);
    ros::spinOnce();

    loop_rate.sleep();

    ROS_INFO("Speed: %.2f Angle: %.2f",speed,angle);


  }


  return 0;
}
