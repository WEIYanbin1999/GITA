# GITA
The official repository of NeurIPS 2024 paper: "GITA: Graph to Image-Text Integration for Vision-Language Graph Reasoning"
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

To reproduce the experimental results, you can run the scripts in the ./Scripts Folder, which includes training and evaluation for all tasks and all models. 

