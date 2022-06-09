#!/bin/sh

export ROS_IP=192.168.0.100 #IP client (Raspberry or Nvidia)
export ROS_MASTER_URI=http://192.168.0.102:11311 # IP Server (Laptop) : 11311
. /home/pablo/Documents/hardware_ws/devel/setup.sh
exec "$@"
