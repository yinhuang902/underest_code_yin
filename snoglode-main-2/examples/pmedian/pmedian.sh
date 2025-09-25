#!/bin/bash
# bash script for running pmedian.py in parallel

mpiexec -np 2 python pmedian.py > output/pmedian_output_parallel.txt