#FROM nvidia/cuda:13.1.0-devel-ubuntu24.04
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG CUDNN_INCLUDE_DIR="/usr/include/x86_64-linux-gnu"

WORKDIR /usr/src/app

# Installing APT packages

RUN apt update && apt install -y software-properties-common
RUN apt update && add-apt-repository ppa:deadsnakes/ppa && apt update
RUN apt upgrade -y
RUN apt install -y python3.13-full python3-virtualenv cython3 ffmpeg
RUN ln -s /usr/bin/python3.13 /usr/bin/python

RUN apt install -y python3-pip
RUN apt install -y git
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libva-dev
RUN apt install -y libavcodec-dev libavformat-dev
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
RUN apt install build-essential cmake git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbbmalloc2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y
RUN apt install -y g++ ant libtbb-dev libgdal-dev 
RUN apt install -y libjpeg-dev
RUN apt install -y libopenjpip-server libopenjp2-7-dev libopenjp2-7

RUN ln -s /usr/lib/x86_64-linux-gnu/libopenjp.so.2.5.0 /usr/lib/x86_64-linux-gnu/libopenjpip.so.2.5.0

# Copying requirements

COPY ./install/requirements.txt ./
COPY ./install/requirements_general_a.txt ./
COPY ./install/requirements_nv.txt ./

# Initializing directories
RUN mkdir ./build
RUN mkdir ./opencv
RUN mkdir ./opencv_contrib

# Copying rest of the requirements and installers
COPY ./install/docker/*.sh ./

# Copying installer fixes
COPY ./install/opencv_fixes/ ./opencv_fixes/

# Initializing git settings
RUN git config --global advice.detachedHead false

# Cloning OpenCV Contrib
RUN git clone -b 4.x --single-branch https://github.com/opencv/opencv_contrib ./opencv_contrib

# Version 4.12 - up to cu129
#RUN cd ./opencv_contrib && git checkout d943e1d61c8bc556a13783e1546ee7c1a9e0b1cf && cd ..

# Version 4.13 - for cu130+
RUN cd ./opencv_contrib && git checkout d99ad2a188210cc35067c2e60076eed7c2442bc3 && cd ..

# Cloning OpenCV
RUN git clone -b 4.x --single-branch https://github.com/opencv/opencv.git ./opencv

#Version 4.12 - up to cu129
#RUN cd ./opencv && git checkout 49486f61fb25722cbcf586b7f4320921d46fb38e && cd ..

# Version 4.13 - for cu130+
RUN cd ./opencv && git checkout fe38fc608f6acb8b68953438a62305d8318f4fcd && cd ..

# Installing general requirements and OpenCV Contrib
RUN bash install_nvidia_docker.sh

# Fixes for OpenCV installation

RUN mkdir -p ./opencv/3rdparty/ippicv/

RUN mkdir -p ./opencv/3rdparty/ippicv/downloads

RUN cp -r ./opencv_fixes/opencv/3rdparty/ippicv ./opencv/3rdparty/

# Installing OpenCV
RUN bash install_nvidia_docker2.sh

# Installiog rest of the NV requirements
RUN bash install_nvidia_docker3.sh

# Copying project files. It is done on this step in order to be able to cache everything above and not recompile OpenCV again in case Python files changed.
COPY ./*.py ./

# Copying models
COPY ./film_models/ ./film_models/

# Cython compilation
RUN bash install_nvidia_docker4.sh

RUN mkdir /share

#ENTRYPOINT ["python", "-W ignore::UserWarning",  "/usr/src/app/main.py"]
ENTRYPOINT ["/usr/src/app/runner.sh"]
