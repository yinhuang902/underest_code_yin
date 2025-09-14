#!/bin/bash
# bash script for running farmer_classic.py in parallel

mpiexec -np 2 python bilinear.py > output/bilinear_output_parallel.txt