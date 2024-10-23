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

Second, specify the tasks in hyper_1 in finetune_lora_loop.sh,

for example, if you want to reproduce the "cycle", you should modify the hyper_1 in finetune_lora_loop.sh as:
~~~
declare -a hyper_1=(
    "cycle"
)
~~~

Third, specify the other hyperparameters, they are arranged in the ordering:
~~~
MODELSIZE
EPOCH  # Epoches from {1,5,10,20,30,50}
BSZ    # per device train batch_size from {16,32}
LORAR  # The rank of the low-rank matrices used in the LoRA adaptation
LORAALPHA  # The scaling factor that controls the magnitude of the low-rank adaptation
MODALTYPE  # Text_Only, Vision_Only, Vision_Text (both image and text)
TASKTYPE  # GVLQA-BASE, GVLQA-AUGET, GVLQA-AUGLY, GVLQA-AUGNO, GVLQA-AUGNS; NODECLS; LINKPRED
UNFREEZEV  # Optional: Fine-tune vision tower or not when Vision_Only or Vision_Text. If True, yes.
LAYOUTAUG  # Optional: Whether to use layout augmentation. If True, yes.
~~~

**!Note**: For each setting, please refer to the following table to find their exact configurations, then use the corresponding configurations to replace the hyper_2 in finetune_lora_loop.sh.

GITA-7B:

|Task|hyper_2: MODELSIZE EPOCH BSZ LORAR LOEAALPHA MODALTYPE TASKTYPE UNFREEZEV LAYOUTAYG|
|---|---|
|connectivity|7b 1 16 128 256 Vision_Text GVLQA-BASE False False|
|cycle|7b 20 16 128 256 Vision_Text GVLQA-BASE False False|
|topology|7b 10 16 128 256 Vision_Text GVLQA-BASE False False|
|shoretest_path|7b 10 16 128 256 Vision_Text GVLQA-BASE False False|
|flow|7b 20 16 128 256 Vision_Text GVLQA-BASE False False|
|matching|7b 5 16 128 256 Vision_Text GVLQA-BASE False False|
|hamilton|7b 30 16 128 256 Vision_Text GVLQA-BASE False False|

GITA-13B:

|Task|hyper_2: MODELSIZE EPOCH BSZ LORAR LOEAALPHA MODALTYPE TASKTYPE UNFREEZEV LAYOUTAYG|
|---|---|
|connectivity|13b 1 16 128 256 Vision_Text GVLQA-BASE False False|
|cycle|13b 10 16 128 256 Vision_Text GVLQA-BASE False False|
|topology|13b 10 16 128 256 Vision_Text GVLQA-BASE False False|
|shoretest_path|13b 10 16 128 256 Vision_Text GVLQA-BASE False False|
|flow|13b 10 16 128 256 Vision_Text GVLQA-BASE False False|
|matching|13b 50 16 128 256 Vision_Text GVLQA-BASE False False|
|hamilton|13b 30 16 128 256 Vision_Text GVLQA-BASE False False|


Finally, run:
~~~
cd GITA
bash ./scripts/train/finetune_lora_loop.sh
~~~

