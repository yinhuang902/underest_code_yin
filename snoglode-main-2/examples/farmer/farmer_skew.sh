#!/bin/bash
# bash script for running farmer_skew.py in parallel

mpiexec -np 3 python farmer_skew.py > output/farmer_skew_output_parallel.txt