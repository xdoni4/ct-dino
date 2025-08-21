# ct-dino

This repo is built on [dinov2](https://github.com/facebookresearch/dinov2) fork that enables running on 3D volumes.

## Installation

### Update submodule
```
git submodule update --init --recursive
cd dinov2 && git checkout 47219105a5181fa4ebc81c1b892a8f0224b61f63
```
### Install dinov2 package
```
conda create -n ct_dino python=3.10 -y && conda activate ct_dino
conda install cuda=12.4 -c nvidia -y
pip install pdm pdm-backend
pdm install --no-isolation
```

## Run

### Openmind
```
torchrun --nproc_per_node 8 --nnodes 1 --master-port 29373  dinov2/dinov2/train/train.py --config-file dinov2/dinov2/configs/train/vitm16_3d_openneuro.yaml --output-dir /your/output/dir
```

#### Convert checkpoint to PrimusM
```
python convertation/convert_pretrain_to_primus_m.py --src /path/to/checkpoint --out /path/to/converted/checkpoint
```