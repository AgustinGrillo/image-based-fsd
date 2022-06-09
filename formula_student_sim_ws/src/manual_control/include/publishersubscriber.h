#ifndef PUBLISHERSUBSCRIBER_H

#define PUBLISHERSUBSCRIBER_H
#include "ros/ros.h"
#include <string>
#include "geometry_msgs/Twist.h"


template<typename PublishT, typename SubscribeT>
class PublisherSubscriber
{
public:
  PublisherSubscriber() {}
  PublisherSubscriber(std::string publishTopicName, std::string subscribeTopicName, int queueSize )
  {
    publisherObject = nH.advertise<PublishT>(publishTopicName,queueSize);
    subscriberObject = nH.subscribe<SubscribeT>(subscribeTopicName,queueSize, &PublisherSubscriber::subscriberCallback,this);
  }
  void subscriberCallback(const geometry_msgs::Twist  msg_rec);

protected:
  ros::Subscriber subscriberObject;
  ros::Publisher publisherObject;
  ros::NodeHandle nH;


};




#endif // PUBLISHERSUBSCRIBER_H
