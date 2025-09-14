#!/bin/bash
# bash script for running farmer_classic.py in parallel

mpiexec -np 3 python farmer_classic.py > output/farmer_classic_output_parallel.txt