#!/bin/bash

python bilinear.py > output/bilinear_output.txt
bash bilinear.sh
jupyter nbconvert --to notebook --inplace --execute bilinear.ipynb