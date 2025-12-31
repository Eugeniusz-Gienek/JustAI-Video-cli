#!/bin/bash
echo "Installation of JustAI Video CLI"
die() { echo "$*" 1>&2 ; exit 1; }
ERRORS_COLLECTED=""
if ! command -v python >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- Python could not be found\n"
fi
if ! command -v cython >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- Cython (not to be confused with Python!) could not be found\n"
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- FFMPEG could not be found\n"
fi
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

if [ -z $CUDNN_INCLUDE_DIR ]; then
ERRORS_COLLECTED="${ERRORS_COLLECTED}- CUDNN could not be found (variable "'$'"CUDNN_INCLUDE_DIR is empty).\n"
fi

if [ ! -z "${ERRORS_COLLECTED}" ]; then
echo "Can't proceed due to following errors found:"
echo -e $ERRORS_COLLECTED
exit 1
fi

NV_VER=$(nvcc --version | grep -oP '^Cuda compilation.*release \K[0-9]+\.[0-9]+')
PY_VER=$(python --version | awk '{print substr($2,0,4)}')
FFMPEG_VER=$(ffmpeg -version | grep version | awk '{ print $3 }')

echo "Python version: " $(python --version)
echo "Cython version: " $(cython --version)
echo "NVCC version: ${NV_VER}"
echo "FFMPEG version: ${FFMPEG_VER}"

if [ ! -d "./venv" ]; then
echo "Creating python virtual environment directory"
python -m venv ./venv || die "Could not create python virtual environment. This might happen if the venv module does not exist in Python installation."
else
echo "Python virtual environment directory already exists. Skipping."
fi

echo "Entering virtual environment."
source venv/bin/activate

echo "Updating pip"
pip install --upgrade pip

# We have to recompile opencv-python because pip-provided version lacks FFMPEG integration which is needed for the project to function.
echo "Installing OpenCV requirements..."

#pip install numpy==2.2.6

pip uninstall -y opencv-python
pip uninstall -y protobuf

if [ "${PY_VER}" == "3.13" ]; then

pip install --no-cache-dir -r requirements_general_a.txt || die "Cannot install general requirements"
pip install protobuf || die "Cannot install Protobuf"
# pip install protobuf==6.32.1 || die "Cannot install Protobuf"
# ==5.29.5 || die "Cannot install Protobuf"
# echo "Installing general requirements..."
# pip install --no-cache-dir -r requirements.txt || die "Cannot install general requirements"
./install_opencv_contrib.sh || die "Cannot install OpenCV Contrib"
echo "Installing OpenCV Contrib..."
bash ./requirements_opencv_3.13.txt || die "Cannot install OpenCV"
pip install --no-cache-dir -r requirements.txt || die "Cannot install general requirements (part 2)"
else

echo "Installing general requirements..."
pip install --no-cache-dir -r requirements.txt || die "Cannot install general requirements"
echo "Installing OpenCV Contrib..."
./install_opencv_contrib.sh || die "Cannot install OpenCV"
bash ./requirements_opencv.txt || die "Cannot install OpenCV"

fi

pip install pyopencl[pocl]

echo "Updating cuda requirements file."

cp requirements_nv.txt requirements_nv2.txt

#if [ "${NV_VER}" == "12.9"  ]; then
##cp requirements_nv.txt requirements_nv2.txt
#else
if [ "${NV_VER}" == "13.0"  ]; then
sed "s/cu129/cu130/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.8"  ]; then
sed "s/cu129/cu128/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.7"  ]; then
sed "s/cu129/cu127/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.6"  ]; then
sed "s/cu129/cu126/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.4"  ]; then
sed "s/cu129/cu124/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
fi; fi; fi; fi; fi

echo "Installing cuda libraries...."
pip install --no-cache-dir -r requirements_nv2.txt || die "Cannot install nvidia requirements"

echo "Installing CLIP requirements..."
bash ./requirements_clip.txt || die "Cannot install CLIP requirements"

echo "Exiting virtual environment."
deactivate
echo "Installation finished."
