#!/bin/bash

dirs=(bilinear farmer ip pmedian quad stochastic_pid)
for dir in "${dirs[@]}"; do
    cd $dir
    bash .run.sh
    cd ..
done