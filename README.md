# AFM-distillation

## Environment Setup

### Allegro

```bash
conda create -n YOUR_ENV_NAME python=3.11 -y
conda activate YOUR_ENV_NAME
pip install nequip
pip install nequip-allegro
pip install -e .
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