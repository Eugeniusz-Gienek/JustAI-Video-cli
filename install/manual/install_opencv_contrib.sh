#!/bin/bash
cd ../
git submodule init
git submodule update
cd opencv_contrib
cmake -DOPENCV_EXTRA_MODULES_PATH=./modules/ -DBUILD_opencv_legacy=OFF ../opencv
make -j5
