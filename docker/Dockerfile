FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

#Upgrade the system
RUN apt clean && apt-get update && apt-get --with-new-pkgs upgrade -y

#Install standart packages
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get update && apt-get install -y --allow-unauthenticated --fix-missing \
    #Install essential tools and libraries
    apt-utils build-essential tree wget curl git vim ctags \
    unzip libopenblas-dev libatlas-base-dev cmake make lsb-release  \
    zlib1g-dev libx11-dev libusb-1.0-0-dev freeglut3-dev \
    sudo ca-certificates gnupg-agent libssl-dev apt-transport-https \
    net-tools openssh-client rsyslog software-properties-common \
    gdb htop valgrind emacs strace ltrace unzip gnuplot gnuplot-qt \
    libjsoncpp-dev libncurses5-dev netcat usbutils libtclap-dev \
    #Install OpenGL related libraries
    libglew-dev glew-utils mesa-utils mesa-utils-extra \
    #Install turbo jpeg
    libturbojpeg libturbojpeg0-dev \
    #Install LAPACK
    libopenblas-dev liblapack-dev liblapacke-dev \
    #Install pip3 and python3 packages
    python3-pip python3-numpy python3-opencv

#Install additional packages
RUN conda remove --force -y ffmpeg && apt-get install -y ffmpeg libavcodec-extra feh mplayer h5utils hdf5-tools libosmesa6

#Install python packages
RUN pip3 install --upgrade pip 
RUN pip3 install numpy matplotlib scipy scikit-build
RUN pip3 install plotly plotly_express
RUN pip3 install tensorboard
RUN pip3 install animatplot
RUN pip3 install pyyaml
RUN pip3 install swig
RUN pip3 install box2d-py
RUN pip3 install roma
RUN pip3 install moderngl
RUN pip3 install pyrr
RUN pip3 install gymnasium pygame pymunk h5py

#Configure NVIDIA hardware acceleration
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

#More python packages
ENV MUJOCO_GL osmesa
RUN pip3 install ogbench
RUN pip3 install torchaug

#Install ROS
RUN \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update
RUN apt-get update && apt-get install -y \
    python3-catkin-tools ros-noetic-ros-base ros-noetic-geometry-msgs \
    ros-noetic-common-msgs ros-noetic-image-geometry ros-noetic-camera-info-manager \
    ros-noetic-image-transport-plugins ros-noetic-compressed-image-transport \
    ros-noetic-image-transport ros-noetic-image-publisher 
RUN pip3 install rospkg rosbags
RUN apt-get install -y ros-noetic-ros-numpy

#Set python cache directory
ENV PYTHONPYCACHEPREFIX /tmp/pycache

