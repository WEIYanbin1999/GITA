# Rendering Graphs for Graph Reasoning in Multimodal Large Language Models

## Install
```bash
conda create -n gitq python=3.10 -y
conda activate gita
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install torch_geometric==2.5.3
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

Please organize the data as follows:
```
├── dataset
│   ├── GITQA-BASE
│   ├── NODECLS
│   ├── LINKPRED
│   └── ...
└── GIT
    └── ...
```