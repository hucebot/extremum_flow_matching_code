#!/bin/bash

IsRunning=`docker ps -f name=extremum_flow_matching | grep -c "extremum_flow_matching"`;
if [ $IsRunning -eq "0" ]; then
    echo "Docker image is not running. Starting it...";
    mkdir -p /tmp/docker_share
    xhost +local:docker
    docker run --rm \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=$XAUTHORITY \
        -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e 'QT_X11_NO_MITSHM=1' \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v /tmp/docker_share:/tmp/docker_share \
        -v `pwd`/../:/workspace \
        --workdir /workspace \
        --ipc host \
        --device /dev/dri \
        --device /dev/snd \
        --device /dev/input \
        --device /dev/bus/usb \
        --privileged \
        --ulimit rtprio=99 \
        --net host \
        --name extremum_flow_matching \
        --entrypoint /bin/bash \
        -ti extremum_flow_matching_docker:latest
else
    echo "Docker image is already running. Opening new terminal...";
    docker exec -ti extremum_flow_matching /bin/bash
fi

