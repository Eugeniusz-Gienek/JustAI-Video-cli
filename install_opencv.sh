#!/bin/bash
#pip install --no-cache-dir opencv-contrib-python
#./install_opencv_contrib.sh
#export CMAKE_ARGS="-D WITH_FFMPEG=ON -D WITH_OPENCL=OFF -D WITH_CUDA=ON -D OPENCV_EXTRA_MODULES_PATH=/%your_path_for_custom_libs%/opencv_contrib/modules"
#export CMAKE_ARGS="-D WITH_FFMPEG=ON -D WITH_OPENCL=OFF"
C_DIR=$(pwd)
export CMAKE_ARGS="-D CMAKE_BUILD_TYPE=Release -D WITH_FFMPEG=ON -D WITH_OPENCL=ON -D WITH_CUDA=ON -D OPENCV_EXTRA_MODULES_PATH=${C_DIR}/opencv_contrib/modules/ -D WITH_NVCUVID=OFF -D WITH_NVCUVENC=OFF  -DLAPACK=OFF -DBLAS=OFF -DINSTALL_CREATE_DISTRIB=OFF -D BUILD_opencv_python3=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF"
# -D WITH_CUDA=ON"
# -DCMAKE_CXX_STANDARD=14 -DLAPACK=OFF"
#pip install --verbose --no-binary opencv-contrib-python --no-deps opencv-contrib-python
#git clone https://github.com/opencv/opencv_contrib.git
#cd opencv_contrib/modules

#pip install --no-cache-dir --verbose --no-binary opencv-python --no-deps opencv-python==4.12.0.88

mkdir -p ./opencv/build
cd ./opencv/

git checkout .
git submodule init
git submodule update
#cmake --target clean
cp ../opencv_fixes/CMakeLists.txt ./CMakeLists.txt
CMAKE_SOURCE_DIR=./ CMAKE_BINARY_DIR=./build cmake -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)")  -D PYTHON_EXECUTABLE=$(which python) ./
make -j$(nproc)
make install
#cmake --clean-first
#cmake --build .

#pip install --no-binary opencv-python-headless --no-deps opencv-python-headless
#make install
