# ct-dino

## Installation

```
conda create -n ct_dino python=3.10 -y && conda activate ct_dino
conda install cuda=12.4 -c nvidia -y
pip install pdm pdm-backend
pdm install --no-isolation
```