#!/bin/bash

python farmer_classic.py > output/farmer_classic_output.txt
bash farmer_classic.sh
python farmer_skew.py > output/farmer_skew_output.txt
bash farmer_skew.sh