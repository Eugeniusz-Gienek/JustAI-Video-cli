#!/bin/sh
rm -f *.c
rm -f *.so
python setup.py build_ext --inplace
