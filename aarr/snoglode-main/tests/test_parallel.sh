#!/bin/bash
mpiexec -np 2 python -m pytest --with-mpi
mpiexec -np 3 python -m pytest --with-mpi