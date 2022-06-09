#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "geometry_msgs/PoseArray.h"
#include "tf/transform_listener.h"
#include <math.h>

/* Very simple implementation of Pure Pursuit Path tracking Algorithm
 * by R. Craig Conlter CMU-RI-TR-92-0
 * Author: Gtorre
 * Assumes path recieived will have node 0 on robots actual position and the first segment will require
 * turning less than 45ª and will stop one "lookAheadDistance" before last node, the path should be smoothed*/

static double controlFrequency = 20; //[hz] frecuency of publishing for node
static double lookAheadDistance = 5 ; //Only parameter for pure pursuit algorithm, smaller meand more responsive, bigger is more stable
static double linearVelocity = 1, curvature;

// These should be inside a class and not be global
bool stop = true; //flag, tells the robot to stop
double posX, posY, posTheta; //Position of the robot updated in main when nessesary by tf
double closestX,closestY,goalX,goalY,goalX_R,goalY_R; //For internal use of PP algorithm, see the functions that use them

geometry_msgs::PoseArray pathArray;
geometry_msgs::Twist velMsg;


// Constrain angle between -pi and pi
double constrainAngle(double x){
  x = fmod(x + M_PI,2*M_PI);
  if (x < 0)
    x += 2*M_PI;
  return x - M_PI;
}

//Subscriber for path
void pathCallback(const geometry_msgs::PoseArrayConstPtr &msg){

  if (msg->poses.size() == 0)
    stop = true;
  else
    stop = false;

  ROS_INFO("I received a path of: [%d] segments", static_cast<int> (msg->poses.size()));
  pathArray.poses = msg->poses;
}

//function to find finds nearest point to the robot on a line described by two points
void findClosestPoint(geometry_msgs::Pose start, geometry_msgs::Pose end){
  double a = end.position.x - start.position.x;
  double b = start.position.y - end.position.y;
  double c = -end.position.x* start.position.y + start.position.x* end.position.y;
  closestX = (a*(a*posX-b*posY)-b*c)/(a*a+b*b);
  closestY = (b*(b*posY-a*posX)-a*c)/(a*a+b*b);
}

//answers: is the asked point close to the robot?
bool closeToNode(double nextPosX, double nextPosY, double distance){
  return (pow(nextPosX-posX,2.0)+pow(nextPosY-posY,2.0)<pow(distance,2.0));
}

//answers: is the asked angle close to the robots angle (in radians)?
bool closeToAngle( double targetAngle, double angleRange){
  return ( fabs(constrainAngle(posTheta-constrainAngle(targetAngle))) < angleRange);
}

//function that finds goal pos in global coords, a point along a line of distance = distance threshold
void findGoalPoint(geometry_msgs::Pose end){
  goalX = closestX;
  goalY = closestY;
  for(int j = 0; j < 251 && closeToNode(goalX,goalY,lookAheadDistance); ++j){
    goalX = 0.004* (closestX * (float) (250-j) + end.position.x * (float) j);
    goalY = 0.004* (closestY * (float) (250-j) + end.position.y * (float) j);
  }
}

//Transform from global to local coords
void calculateGoalRobotCoords() {
  goalX_R = cos(posTheta) * (goalX - posX) + sin(posTheta) * (goalY - posY);
  goalY_R = -sin(posTheta) * (goalX - posX) + cos(posTheta) * (goalY - posY);
}

//Calculates curvature, used for Pure Pursuit Algorithm
void calculateRotationCurvature() {
  curvature = (2.0 * goalY_R) / (lookAheadDistance * lookAheadDistance);
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "driver");
  ros::NodeHandle nh("~"); //~ private node Handle
  ros::Rate loop_rate(controlFrequency);

  tf::TransformListener listener;
  tf::StampedTransform transform;
  
  ros::Publisher velPublisher = nh.advertise<geometry_msgs::Twist>("/fs/motor_controller/cmd_vel", 1);
  ros::Subscriber pathSubscriber = nh.subscribe("/fs/path", 5, &pathCallback);

  while (ros::ok() && nh.ok())
  {
    //Idle
    ROS_INFO("Waiting for driving instructions");
    while (ros::ok() && nh.ok()&& stop)
    {
      velMsg.angular.z = 0.0;
      velMsg.linear.x = 0.0;
      velPublisher.publish(velMsg);

      ros::spinOnce();
      loop_rate.sleep();
    }

    //Driving over path
    ROS_INFO("Received driving instructions");
    for(unsigned long i= 0;ros::ok() && nh.ok() && !stop && i< pathArray.poses.size()-1; ++i)
    {
      ROS_INFO("im at node:%lu  X:%f Y:%f , pos: X:%f Y:%f",i,pathArray.poses[i].position.x,pathArray.poses[i].position.y,posX,posY);
      while(ros::ok() && nh.ok() && !stop && !closeToNode(pathArray.poses[i+1].position.x,pathArray.poses[i+1].position.y,lookAheadDistance))
      {

        try {
          listener.lookupTransform("/odom","/base_link",  ros::Time(0), transform);
          posX = transform.getOrigin().getX();
          posY = -transform.getOrigin().getY(); //Adjust signs according to your TF
          posTheta = constrainAngle(-tf::getYaw(transform.getRotation()));//for child frames this "-" might not be nesesary
          //ROS_INFO("X: %f , Y: %f , tita: %f", posX, posY, posTheta);
        } catch (tf::TransformException &ex) {
          ROS_ERROR("%s", ex.what());
          ros::Duration(1.0).sleep();
        }

        //Pure pursuit algorithm
        findClosestPoint(pathArray.poses[i],pathArray.poses[i+1]);
        findGoalPoint(pathArray.poses[i+1]);
        calculateGoalRobotCoords();
        calculateRotationCurvature();
        velMsg.linear.x = linearVelocity;
        velMsg.angular.z = linearVelocity * curvature;
        velPublisher.publish(velMsg);

        ros::spinOnce();
        loop_rate.sleep();
      }
    }
    stop = true;
  }
  return 0;
}
