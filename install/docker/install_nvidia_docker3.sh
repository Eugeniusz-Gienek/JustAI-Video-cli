#!/bin/bash
echo "Installation of JustAI Video CLI (part 3)"

die() { echo "$*" 1>&2 ; exit 1; }

SUPPORTED_PY_VER="3.13"

# Version of cuda in requirements_nv.txt file (cuXXX)
ORIG_NV_VER="129"

NV_VER=$(nvcc --version | grep -oP '^Cuda compilation.*release \K[0-9]+\.[0-9]+')
PY_VER=$(python3.13 --version | awk '{print substr($2,0,4)}')
FFMPEG_VER=$(ffmpeg -version | grep version | awk '{ print $3 }')

C_DIR=$(pwd)

echo "Entering virtual environment."

. ./venv/bin/activate || die "Could not enter virtual environment"

# We have to recompile opencv-python because pip-provided version lacks FFMPEG integration which is needed for the project to function.

echo "Updating CUDA requirements file."

cp requirements_nv.txt requirements_nv2.txt

# This if-case is for the future - in order to set the correct version automatically. Basically it is mapping.

if [ "${NV_VER}" == "13.1"  ]; then
sed "s/cu${ORIG_NV_VER}/cu131/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "13.0"  ]; then
sed "s/cu${ORIG_NV_VER}/cu130/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.8"  ]; then
sed "s/cu${ORIG_NV_VER}/cu128/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.7"  ]; then
sed "s/cu${ORIG_NV_VER}/cu127/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.6"  ]; then
sed "s/cu${ORIG_NV_VER}/cu126/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
else
if [ "${NV_VER}" == "12.4"  ]; then
sed "s/cu${ORIG_NV_VER}/cu124/g" -i requirements_nv2.txt || die "Could not update requirements_nv2.txt file."
fi; fi; fi; fi; fi; fi

echo "Installing CUDA libraries...."
pip install --no-cache-dir -r ./requirements_nv2.txt || die "Cannot install nvidia requirements"

echo "Installing CLIP requirements..."
bash "${C_DIR}/install_clip_docker.sh" || die "Cannot install CLIP requirements"

echo "Exiting virtual environment."
deactivate

echo "Dependencies installation finished."

echo "Copying models..."
