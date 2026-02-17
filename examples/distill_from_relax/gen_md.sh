#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate mattersim-tune

python synthetic_gen_relax.py \
    --model_path "MatterSim-v1.0.0-1M" \
    --starting_structrues "../distill_from_md/data/h2o_1593_train_25.xyz" \
    --device 0 \
    --n_per_structure 4 

python synthetic_gen_relax.py \
    --model_path "MatterSim-v1.0.0-1M" \
    --starting_structrues "../distill_from_md/data/h2o_1593_train_25.xyz" \
    --device 0 \
    --n_per_structure 40

python synthetic_gen_relax.py \
    --model_path "MatterSim-v1.0.0-1M" \
    --starting_structrues "../distill_from_md/data/h2o_1593_train_25.xyz" \
    --device 0 \
    --n_per_structure 400