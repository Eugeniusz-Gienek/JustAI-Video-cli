#!/bin/bash
git submodule init
git submodule update
#git submodule update --remote
cd opencv_contrib
#git checkout origin/4.x
#git config -f .gitmodules submodule.opencv_contrib.branch 4.x
cmake -DOPENCV_EXTRA_MODULES_PATH=./modules/ -DBUILD_opencv_legacy=OFF ../opencv
#cmake -DOPENCV_EXTRA_MODULES_PATH=./modules -DBUILD_opencv_legacy=OFF ../opencv
make -j5
#make install
