from __future__ import annotations

from ase.io import read
from afm_distill.models.allegro.util import nequip_model_package


atoms = read("./data/li3po4_192.xyz")
output_path = nequip_model_package(
    ckpt_path="./checkpoints/allegro-4.0A-T=1.ckpt",
    output_path="./checkpoints/allegro-4.0A-T=1.nequip.zip",
    atoms_example=atoms
)
"""
nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pt2 \
  --device [cpu|cuda] \
  --mode aotinductor \
  --target [ase|pair_nequip|pair_allegro|...]

nequip-compile \
    ./checkpoints/allegro-4.0A-T=1.nequip.zip \
    ./checkpoints/allegro-4A-T=1-aot-lammps.nequip.pt2 \
    --device cuda \
    --mode aotinductor \
    --target ase
    
nequip-compile \
    ./checkpoints/allegro-4.0A-T=1.nequip.zip \
    ./checkpoints/allegro-4.0A-T=1.nequip.pth \
    --device cuda \
    --mode torchscript \
    --target ase
    
    
nequip-compile \
    ./checkpoints/allegro-4.0A-T=1.nequip.zip \
    ./checkpoints/allegro-4A-T=1-aot-lammps.nequip.pt2 \
    --device cuda \
    --mode aotinductor \
    --target pair_allegro
"""
