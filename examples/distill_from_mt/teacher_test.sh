#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

# conda activate mattersim-tune
# python teacher_test.py --model_path "MatterSim-v1.0.0-1M"
# python teacher_test.py --model_path "MatterSim-v1.0.0-1M" --dataset_path "/nethome/lkong88/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz"

conda activate uma-tune
python teacher_test.py --model_path "uma-s-1p1" --task_name "omat"
python teacher_test.py --model_path "uma-s-1p1" --task_name "omat" --dataset_path "/nethome/lkong88/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz"

# conda activate nequip-tune
# python teacher_test.py --model_path "NequIP-OAM-L-0.1"
# python teacher_test.py --model_path "NequIP-OAM-L-0.1" --dataset_path "/nethome/lkong88/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz"

# conda activate mace-tune
# python teacher_test.py --model_path "mace-medium-omat-0"
# python teacher_test.py --model_path "mace-medium-omat-0" --dataset_path "/nethome/lkong88/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz"

# conda activate orbv3-tune
# python teacher_test.py --model_path "orb-v3-conservative-inf-omat"
# python teacher_test.py --model_path "orb-v3-conservative-inf-omat" --dataset_path "/nethome/lkong88/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz"

