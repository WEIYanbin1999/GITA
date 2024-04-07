# Rendering Graphs for Graph Reasoning in Multimodal Large Language Models

## Install
```bash
conda create -n gitqa python=3.10 -y
conda activate gitqa
pip install --upgrade pip  # enable PEP 660 support
pip install "fschat[model_worker,webui]"
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
