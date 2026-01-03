#!/bin/bash
#pip install --no-cache-dir opencv-contrib-python
#./install_opencv_contrib.sh
#export CMAKE_ARGS="-D WITH_FFMPEG=ON -D WITH_OPENCL=OFF -D WITH_CUDA=ON -D OPENCV_EXTRA_MODULES_PATH=/%your_path_for_custom_libs%/opencv_contrib/modules"
#export CMAKE_ARGS="-D WITH_FFMPEG=ON -D WITH_OPENCL=OFF"

cd ..

C_DIR=$(pwd)
C_PYTHON3_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")
C_PYTHON3_PACKAGES_PATH=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
C_CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)")
C_PYTHON_EX=$(which python)

export CMAKE_ARGS="\
-D CMAKE_BUILD_TYPE=Release \
-D WITH_FFMPEG=ON \
-D WITH_OPENCL=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D BUILD_opencv_cudacodec=ON \
-D OPENCV_EXTRA_MODULES_PATH=${C_DIR}/../opencv_contrib/modules/ \
-D WITH_NVCUVID=OFF \
-D WITH_NVCUVENC=OFF \
-DLAPACK=OFF \
-DBLAS=OFF \
-DINSTALL_CREATE_DISTRIB=OFF \
-D BUILD_opencv_python3=ON \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D PYTHON3_INCLUDE_DIR=${C_PYTHON3_INCLUDE_DIR} \
-D PYTHON3_PACKAGES_PATH=${C_PYTHON3_PACKAGES_PATH} \
-D CMAKE_SOURCE_DIR=./ \
-D CMAKE_BINARY_DIR=./build \
-D CMAKE_INSTALL_PREFIX=${C_CMAKE_INSTALL_PREFIX}\
 -D PYTHON_EXECUTABLE=${C_PYTHON_EX}\
"

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

touch ./3rdparty/ippicv/downloader.cmake

#ls /usr/src/app/opencv/3rdparty/ippicv/ippicv_lnx/iw && exit 1

echo
echo "CMAKE"
echo

cmake $CMAKE_ARGS -S . -B build/

cd build
echo
echo "MAKE"
echo
make -j$(nproc)
echo
echo "MAKE INSTALL"
echo
make install
cd ..

#cmake --clean-first
#cmake --build .

#pip install --no-binary opencv-python-headless --no-deps opencv-python-headless
#make install
