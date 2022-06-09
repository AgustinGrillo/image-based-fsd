#!/bin/sh

export ROS_IP=192.168.0.102 #IP server (Laptop)
export ROS_MASTER_URI=http://192.168.0.102:11311  #IP server:11311
. /home/agus/Documents/Proyectos/hardware_interface/devel/setup.sh #Source workspace
exec "$@"
