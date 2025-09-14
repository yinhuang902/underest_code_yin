#!/bin/bash
# bash script for running pmedian.py in parallel

mpiexec -np 3 python quad.py > output/quad_output_parallel.txt