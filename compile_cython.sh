#!/bin/sh
rm -f *.c
rm -f *.so
python compile_cython.py build_ext --inplace
