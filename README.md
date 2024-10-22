# GITA
[NeurIPS 2024] GITA: Graph to Image-Text Integration for Vision-Language Graph Reasoning

## Please fell free to use our GVLQA Datasets for vision-language reasoning!!
Download:
https://huggingface.co/collections/Yanbin99/gvlqa-datasets-65c705c9488606617e246bd3


## Install
```bash

conda create -n gita python=3.10 -y
conda activate gita
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install torch_geometric==2.5.3
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
sudo apt install graphviz
```

## File structures
Please organize the data as follows:
```
├── dataset
│   ├── GVLQA-BASE
|   ├── GVLQA-AUGET
|   ├── GVLQA-AUGLY
|   ├── GVLQA-AUGNO
|   ├── GVLQA-AUGNS
│   ├── NODECLS
│   ├── LINKPRED
│   └── ...(any custom datasets, applying GITA on existing graph data to generate their vision-language version)
└── GITA
    ├── answer
    ├── checkpoints
    ├── fastchat
    ├── llava
    └── scripts

```

## Reproduction
To reproduce the experimental results, you can run the scripts in the ./Scripts Folder, which includes training and evaluation scripts. 
The detailed configuration in scripts for settings:
