# AFM-distillation

## Environment Setup

### Allegro

```bash
conda create -n YOUR_ENV_NAME python=3.11 -y
conda activate YOUR_ENV_NAME
conda install cmake openmpi -y
conda install -c nvidia cuda-toolkit=12.4 -y
pip install --no-cache-dir torch==2.6 --index-url https://download.pytorch.org/whl/cu124
pip install nequip
pip install nequip-allegro
pip install -e .
```

#### LAMMPS

```bash
make -C src no-all purge

rm -rf ./build/

export CONDA_BUILD_SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"

cmake -S cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DPKG_ML-PACE=ON \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DNEQUIP_AOT_COMPILE=ON \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX/targets/x86_64-linux" \
  -DMKL_INCLUDE_DIR=/tmp \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`

cmake --build build --parallel 16

echo 'export PATH=/path/to/lammps/build:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### SchNet & PaiNN

```bash
conda create -n YOUR_ENV_NAME python=3.11 -y
conda activate YOUR_ENV_NAME
pip install schnetpack
pip install -e .
```
<!-- 
### CACE -->

## Update Configs

> This section is for developer. Each time after a major updates, developers should run this command under the root dir of this repo to update configs correspondingly. 

```bash
rm -rf ./src/afm_distill/configs/ && nshconfig-export -o ./src/afm_distill/configs afm_distill
```