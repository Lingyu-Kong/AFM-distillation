#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


conda activate allegro_lammps

python allegro_distill.py --devices 0 1 2 3 4 5 6 7

# python allegro_baseline.py --devices 0 1 2 3

conda activate schnet_distill

python painn_distill.py --devices 0 1 2 3 4 5 6 7

# python painn_baseline.py --devices 0 1 2 3