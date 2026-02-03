#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate mattersim-tune

python synthetic_gen_md.py --model_path "/nethome/lkong88/MatterTune/examples/water-thermodynamics/checkpoints/mattersim-1m-best-30-refill-conservative.ckpt"

python synthetic_gen_md.py --model_path "MatterSim-v1.0.0-1M"