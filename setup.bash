#!/usr/bin/env bash

#Setup Python path environment variable
#Only append the source directory if it does not already exist
current_path=`pwd`
[[ ":${PYTHONPATH}:" != *":${current_path}:"* ]] && export PYTHONPATH="${PYTHONPATH}:${current_path}"

#Setup ROS environment
source /opt/ros/noetic/setup.bash

