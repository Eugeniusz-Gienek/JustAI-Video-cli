#!/bin/bash
echo "Installation of JustAI Video CLI"

die() { echo "$*" 1>&2 ; exit 1; }

SUPPORTED_PY_VER="3.13" # we'll not be limiting to this version only though. 3.12 worked just fine.
PY_VER=""
ERRORS_COLLECTED=""

# Checking for Python
if ! command -v python >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- Python could not be found\n"
else
PY_VER=$(python --version | awk '{print substr($2,0,4)}')
fi

# Checking for Cython
if ! command -v cython >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- Cython (not to be confused with Python!) could not be found\n"
fi

# Checking for FFMPEG
if ! command -v ffmpeg >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- FFMPEG could not be found\n"
fi

# Checking for CUDA
if ! command -v nvcc >/dev/null 2>&1; then
FXPATH="/opt/cuda/bin/:/usr/local/cuda/bin"
PATH="${PATH}:${FXPATH}"
if ! command -v nvcc >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- Nvidia Cuda toolkit (nvcc) could not be found. This might happen if cuda's bin path is not in "'$PATH'" variable\n"
else
echo
echo "!!!!! ATTENTION !!!!!"
echo
echo "Cuda's bin path should be added to PATH variable, temporary fix for the installer was successful though so proceeding."
echo "Consider adding the '${FXPATH}' to your system PATH,"
echo "for example by adding this in .bashrc: export PATH="'"$PATH:'"${FXPATH}'"
echo
echo "!!!!!!!!!!!!!!!!!!!!!"
echo
fi
fi

# Checking for cuDNN
if [ -z $CUDNN_INCLUDE_DIR ]; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- CUDNN could not be found (variable "'$'"CUDNN_INCLUDE_DIR is empty).\n"
fi

## Checking Python version
#if [ "${PY_VER}" != "${SUPPORTED_PY_VER}" ]; then
#ERRORS_COLLECTED="${ERRORS_COLLECTED}- Python version is not set to ${SUPPORTED_PY_VER}\n"
#fi

if [ ! -z "${ERRORS_COLLECTED}" ]; then
echo "Can't proceed due to following errors found:"
echo -e $ERRORS_COLLECTED
exit 1
fi

NV_VER=$(nvcc --version | grep -oP '^Cuda compilation.*release \K[0-9]+\.[0-9]+')
FFMPEG_VER=$(ffmpeg -version | grep version | awk '{ print $3 }')

# Showing found info

echo "Python version: " $(python --version)
echo "Cython version: " $(cython --version)
echo "NVCC version: ${NV_VER}"
echo "FFMPEG version: ${FFMPEG_VER}"

# Creating virtual environment
if [ ! -d "../../venv" ]; then
echo "Creating python virtual environment directory"
python -m venv ../../venv || die "Could not create python virtual environment. This might happen if the venv module does not exist in Python installation."
else
echo "Python virtual environment directory already exists. Skipping."
fi

C_DIR=$(pwd)

echo "Entering virtual environment."

. ../../venv/bin/activate || die "Could not enter virtual environment"

# We have to recompile opencv-python because pip-provided version lacks FFMPEG integration which is needed for the project to function.
echo "Installing OpenCV requirements..."

# We will have to recompile OpenCV in order to have included FFMPEG in it (default PIP version does NOT include it and it is essential that it does)
pip uninstall -y opencv-python
pip uninstall -y protobuf

# There might be errors related to cached cmake files so removing these files in order to overcome this issue.
rm -f ../opencv_contrib/CMakeCache.txt
rm -f ../opencv/CMakeCache.txt

if [ "${PY_VER}" == "3.13" ]; then

# Installing general requirements, not yet including NVIDIA PIP repo
pip install --no-cache-dir -r ../requirements_general_a.txt || die "Cannot install general requirements"
pip install protobuf || die "Cannot install Protobuf"

# Installing OpenCV Contrib modules for CUDA to work
echo "Installing OpenCV Contrib..."
./install_opencv_contrib.sh || die "Cannot install OpenCV Contrib"

pip install pyopencl[pocl]

mkdir -p ../opencv/3rdparty/ippicv/
mkdir -p ../opencv/3rdparty/ippicv/downloads
cp -r ../opencv_fixes/opencv/3rdparty/ippicv ../opencv/3rdparty/

rm -f ../opencv/CMakeCache.txt

echo "Installing OpenCV Contrib..."
./install_opencv.sh || die "Cannot install OpenCV"
pip install --no-cache-dir -r ../requirements.txt || die "Cannot install general requirements (part 2)"
else

echo "Installing general requirements..."
pip install --no-cache-dir -r ../requirements.txt || die "Cannot install general requirements"
echo "Installing OpenCV Contrib..."

./install_opencv_contrib.sh || die "Cannot install OpenCV"
pip install pyopencl[pocl]

mkdir -p ../opencv/3rdparty/ippicv/
mkdir -p ../opencv/3rdparty/ippicv/downloads
cp -r ../opencv_fixes/opencv/3rdparty/ippicv ../opencv/3rdparty/

rm -f ../opencv/CMakeCache.txt
./install_opencv.sh || die "Cannot install OpenCV"
fi

pip install pyopencl[pocl]

echo "Updating CUDA requirements file."

cp ../requirements_nv.txt ../requirements_nv2.txt

if [ "${NV_VER}" == "13.1"  ]; then
sed "s/cu129/cu131/g" -i ../requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "13.0"  ]; then
sed "s/cu129/cu130/g" -i ../requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.8"  ]; then
sed "s/cu129/cu128/g" -i ../requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.7"  ]; then
sed "s/cu129/cu127/g" -i ../requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.6"  ]; then
sed "s/cu129/cu126/g" -i ../requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.4"  ]; then
sed "s/cu129/cu124/g" -i ../requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
fi; fi; fi; fi; fi; fi

echo "Installing CUDA libraries...."
pip install --no-cache-dir -r ../requirements_nv2.txt || die "Cannot install nvidia requirements"

echo "Installing CLIP requirements..."
bash ./install_clip.sh || die "Cannot install CLIP requirements"

echo "Compiling Cython...."
../../compile_cython.sh

echo "Exiting virtual environment."
deactivate

echo "Installation complete."
