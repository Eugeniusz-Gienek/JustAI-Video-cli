#!/bin/bash
# Second part (Docker caching purposes)

echo "Installation of JustAI Video CLI part 2"

die() { echo "$*" 1>&2 ; exit 1; }

SUPPORTED_PY_VER="3.13"

NV_VER=$(nvcc --version | grep -oP '^Cuda compilation.*release \K[0-9]+\.[0-9]+')
PY_VER=$(python3.13 --version | awk '{print substr($2,0,4)}')
FFMPEG_VER=$(ffmpeg -version | grep version | awk '{ print $3 }')

C_DIR=$(pwd)

echo "Entering virtual environment."

. ../../venv/bin/activate || die "Could not enter virtual environment"

#pwd

# We have to recompile opencv-python because pip-provided version lacks FFMPEG integration which is needed for the project to function.
echo "Installing OpenCV requirements..."

if [ "${PY_VER}" == "${SUPPORTED_PY_VER}" ]; then
# Clearing cache just in case
rm -f ../opencv/CMakeCache.txt

# Installing OpenCV itself.
echo "Installing OpenCV from source with FFMPEG..."
./install_opencv_docker.sh || die "Cannot install OpenCV"

echo "Installing rest of the requirements"
# Installing rest of the requirements in case we forgot smth or it became accidentally uninstalled by PIP.
pip install --no-cache-dir -r ../requirements.txt || die "Cannot install general requirements (part 2)"

else
echo "Unsupported python version (expected ${SUPPORTED_PY_VER}, got ${PY_VER})"
exit 1
fi
