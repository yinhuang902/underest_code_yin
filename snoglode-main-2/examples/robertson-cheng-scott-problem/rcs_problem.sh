#!/bin/bash
# bash script for running problem.py in parallel

mpiexec -np 2 python rcs_problem.py > output/rcs_problem_output_parallel.txt