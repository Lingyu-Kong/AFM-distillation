#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate schnet_distill_new
python md_speed_test.py --model ./checkpoints/schnet-5.0A-T=3.ckpt --nl_fn_type "vesin" --skin_cutoff 1.0


conda activate allegro_distill_new
python md_speed_test.py --model ./checkpoints/allegro-4.0A-T=1.ckpt