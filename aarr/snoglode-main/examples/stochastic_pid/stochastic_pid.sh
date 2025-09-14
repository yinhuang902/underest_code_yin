#!/bin/bash
# bash script for running pmedian.py in parallel

mpiexec -np 3 python stochastic_pid.py > output/stochastic_pid_output_parallel.txt