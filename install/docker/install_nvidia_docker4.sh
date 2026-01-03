#!/bin/bash
echo "Cython compiling..."

die() { echo "$*" 1>&2 ; exit 1; }

echo "Entering virtual environment."

. ./venv/bin/activate || die "Could not enter virtual environment"

./compile_cython.sh

echo "Installation complete."
