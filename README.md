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

Training:  
For each setting, before fine-tuning, you should modify the hyperparameters in the finetuning script finetune_lora_loop.sh with following configurations:

First, specify the gpu_ids as the ids of the GPUs you want to use:
~~~
gpu_ids=(
    "0,1,2,3,4,5,6,7"
)
~~~
If you use 8 GPUs from 0 to 7, or if you want to use a single GPU:
~~~
gpu_ids=(
    "0"
)
~~~

Second, specify the tasks, for example, you want to reproduce the "cycle"
~~~
declare -a hyper_1=(
    "cycle"
)
~~~

Third, specify the other hyperparameters, they are arranged in ordering "
MODELSIZE
EPOCH
BSZ
LORAR  # The rank of the low-rank matrices used in the LoRA adaptation (default: 64)
LORAALPHA  # The scaling factor that controls the magnitude of the low-rank adaptation (default: 16)
MODALTYPE  # Text_Only, Vision_Only, Vision_Text (both image and text)
TASKTYPE  # GVLQA-BASE, GVLQA-AUGET, GVLQA-AUGLY, GVLQA-AUGNO, GVLQA-AUGNS; NODECLS; LINKPRED
UNFREEZEV  # Optional: Fine-tune vision tower or not when Vision_Only or Vision_Text. If True, yes. (default: True)
LAYOUTAUG  # Optional: Whether use layout augmentation. If True, yes. (default: False)
"
For each setting, please refer following table to find their exact configurations to modify the hyper_2 in finetune_lora_loop.sh
GITA-7B:
|Task|hyper_2|
|---|---|
|cycle||
GITA-13B:


Finally, run:
~~~
cd GITA
bash ./scripts/train/finetune_lora_loop.sh
~~~

