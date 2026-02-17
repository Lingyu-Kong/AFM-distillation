#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate mattersim-tune

# python synthetic_gen_md.py \
#     --model_path "/nethome/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/md_0.xyz" \
#     --device 0

# python synthetic_gen_md.py \
#     --model_path "/nethome/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/md_1.xyz" \
#     --device 1

# python synthetic_gen_md.py \
#     --model_path "/nethome/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/md_2.xyz" \
#     --device 2

# python synthetic_gen_md.py \
#     --model_path "/nethome/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/md_3.xyz" \
#     --device 3


# python synthetic_gen_md.py --model_path "MatterSim-v1.0.0-1M" \
#     --starting_structrues "./data/water_30_0.xyz" \
#     --device 0

# python synthetic_gen_md.py --model_path "MatterSim-v1.0.0-1M" \
#     --starting_structrues "./data/water_30_1.xyz" \
#     --device 1

# python synthetic_gen_md.py --model_path "MatterSim-v1.0.0-1M" \
#     --starting_structrues "./data/water_30_2.xyz" \
#     --device 2

# python synthetic_gen_md.py --model_path "MatterSim-v1.0.0-1M" \
#     --starting_structrues "./data/water_30_3.xyz" \
#     --device 3

# python synthetic_gen_relax.py \
#     --model_path "/home/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/h2o_1593_train_25.xyz" \
#     --device 0 \
#     --n_per_structure 4 

# python synthetic_gen_relax.py \
#     --model_path "/home/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/h2o_1593_train_25.xyz" \
#     --device 0 \
#     --n_per_structure 40

# python synthetic_gen_relax.py \
#     --model_path "/home/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
#     --starting_structrues "./data/h2o_1593_train_25.xyz" \
#     --device 0 \
#     --n_per_structure 400

python synthetic_gen_relax.py \
    --model_path "/home/lkong88/MatterTune/examples/hidden/ensemble_test/checkpoints/mattersim-1m-best.ckpt" \
    --starting_structrues "./data/h2o_1593_train_25.xyz" \
    --device 0 \
    --n_per_structure 800