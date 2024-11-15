# 🌟 GITA: Graph to Image-Text Integration for Vision-Language Graph Reasoning 🌟

Welcome to the forefront of vision-language graph reasoning! We're thrilled to introduce this promising new topic that connects the VLM, reasoning, and graph communities. Dive in to explore our pioneering contributions!

## 🚀 Contribution 1: GVLQA Benchmark 🚀

Introducing **GVLQA Benchmark**, the **first-ever** vision-language reasoning benchmark designed for general graph reasoning. This is a monumental step forward in the field! 🎉

🔗 **Download Now**: Access the GVLQA datasets from our [Hugging Face Collection](https://huggingface.co/collections/Yanbin99/gvlqa-datasets-65c705c9488606617e246bd3).

## 🤖 Contribution 2: GITA 7B/13B 🤖

Introducing **GITA-7B/13B**, a groundbreaking series of Vision-Language Models crafted specifically for vision-language graph reasoning. These models are expertly fine-tuned on the GVLQA datasets using the powerful LLaVA-1.5 backbone. 🎉

### GITA-7B/13B are Pre-Trained Vision-Language Models with Graph Structural Understanding

**GITA-7B/13B** are pre-trained vision-language models uniquely equipped with graph structural understanding. Their ability to perceive and process graph structures distinguishes them as a robust starting point for any project requiring advanced graph reasoning capabilities.

### Model Zoo

We include the finetuned weights of GITA-7B/13B (LoRa Adaptor and projector) in the `checkpoints/Vision_Text/GVLQA_BASE` directory, they should be used together with LLaVA-v1.5 as we did in `/llava/custom_eval/eval.py` line 201 where invoke the method `load_pretrained_model`. 

To conveniently use GITA-7B/13B as **pre-trained models for downstream graph problems**, we also offer the packed version, where all weights from both the GITA modifications and the original LLaVA weights are packed into a single comprehensive model. Explore our Model Zoo for seamless access:

- **GITA-7B**: [Hugging Face - GITA-7B](https://huggingface.co/Yanbin99/GITA-7B)
- **GITA-13B**: [Hugging Face - GITA-13B](https://huggingface.co/Yanbin99/GITA-13B)


## 🛠️ Install

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

## 📂 File Structure

Please organize the data as follows:

```
├── local_llm
│   ├── llava-v1.5-7b
│   ├── llava-v1.5-13b
│   ├── vicuna-v1.5-7b
│   ├── vicuna-v1.5-13b
├── dataset
│   ├── GVLQA-BASE
│   ├── GVLQA-AUGET
│   ├── GVLQA-AUGLY
│   ├── GVLQA-AUGNO
│   ├── GVLQA-AUGNS
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

## 🔄 Reproduction

Before reproduction, please download the GVLQA datasets from [Hugging Face](https://huggingface.co/collections/Yanbin99/gvlqa-datasets-65c705c9488606617e246bd3). If you do not want to use visual-graph-based augmentations, downloading GVLQA-BASE is sufficient.

### 🏋️‍♂️ Training

To reproduce the experimental results, you can run the scripts in the `./scripts` folder, which includes training and evaluation scripts.

#### Step 1: GPU Configuration

Specify the `gpu_ids` in `finetune_lora_loop.sh`:

```bash
gpu_ids=(
    "0,1,2,3,4,5,6,7"
)
```

For a single GPU:

```bash
gpu_ids=(
    "0"
)
```

#### Step 2: Task Specification

Modify `hyper_1` in `finetune_lora_loop.sh`:

Example for "cycle":

```bash
declare -a hyper_1=(
    "cycle"
)
```

#### Step 3: Hyperparameter Configuration

Specify the hyperparameters in `hyper_2`:

```bash
MODELSIZE
EPOCH  # Epoches from {1,5,10,20,30,50}
BSZ    # per-device train batch size from {16,32}. However, due to the use of gradient accumulation, the actual total batch size remains constant at 128, regardless of the specific per-device batch size value chosen.
LORAR  # The rank of the low-rank matrices used in the LoRA adaptation
LORAALPHA  # The scaling factor that controls the magnitude of the low-rank adaptation
MODALTYPE  # Text_Only, Vision_Only, Vision_Text (both image and text)
TASKTYPE  # GVLQA-BASE, GVLQA-AUGET, GVLQA-AUGLY, GVLQA-AUGNO, GVLQA-AUGNS; NODECLS; LINKPRED
UNFREEZEV  # Optional: Fine-tune vision tower or not when Vision_Only or Vision_Text. If True, yes.
LAYOUTAUG  # Optional: Whether to use layout augmentation online. If True, yes.
```

Refer to the following tables for exact configurations:

**GITA-7B:**

| Task           | hyper_2 Configuration                                                                 |
|:---------------|:--------------------------------------------------------------------------------------|
| connectivity   | 7b 1 16 128 256 Vision_Text GVLQA-BASE False False                                    |
| cycle          | 7b 20 16 128 256 Vision_Text GVLQA-BASE False False                                   |
| topology       | 7b 10 16 128 256 Vision_Text GVLQA-BASE False False                                   |
| shortest_path  | 7b 10 16 128 256 Vision_Text GVLQA-BASE False False                                   |
| flow           | 7b 20 16 128 256 Vision_Text GVLQA-BASE False False                                   |
| matching       | 7b 5 16 128 256 Vision_Text GVLQA-BASE False False                                    |
| hamilton       | 7b 30 16 128 256 Vision_Text GVLQA-BASE False False                                   |

**GITA-13B:**

| Task           | hyper_2 Configuration                                                                 |
|:---------------|:--------------------------------------------------------------------------------------|
| connectivity   | 13b 1 16 128 256 Vision_Text GVLQA-BASE False False                                   |
| cycle          | 13b 10 16 128 256 Vision_Text GVLQA-BASE False False                                  |
| topology       | 13b 10 16 128 256 Vision_Text GVLQA-BASE False False                                  |
| shortest_path  | 13b 10 16 128 256 Vision_Text GVLQA-BASE False False                                  |
| flow           | 13b 10 16 128 256 Vision_Text GVLQA-BASE False False                                  |
| matching       | 13b 50 16 128 256 Vision_Text GVLQA-BASE False False                                  |
| hamilton       | 13b 30 16 128 256 Vision_Text GVLQA-BASE False False                                  |

**GITA-7B on GVLQA-AUGLY (Layout Augmentation):**

| Task           | hyper_2 Configuration                                                                 |
|:---------------|:--------------------------------------------------------------------------------------|
| connectivity   | 7b 10 16 64 16 Vision_Only GVLQA-AUGLY True False                                     |
| cycle          | 7b 10 16 128 256 Vision_Only GVLQA-AUGLY False False                                  |
| topology       | 7b 1 16 128 256 Vision_Only GVLQA-AUGLY False False                                   |
| shortest_path  | 7b 20 16 128 256 Vision_Only GVLQA-AUGLY False False                                  |
| flow           | 7b 1 16 64 16 Vision_Only GVLQA-AUGLY True False                                      |
| matching       | 7b 20 16 128 256 Vision_Only GVLQA-AUGLY False False                                  |
| hamilton       | 7b 30 16 64 16 Vision_Only GVLQA-AUGLY False False                                    |

**Visual Graph Augmentation Variants (AUGNO, AUGNS, AUGET) and Modality Variants (Vision-Only):**

- Replace `GVLQA-Base` with `GVLQA_AUGLY/GVLQA_AUGNO/GVLQA_AUGNS/GVLQA_AUGET`
- Replace `Vision_Text` with `Vision_Only`

#### Step 4: Execute Training

```bash
cd GITA
bash ./scripts/train/finetune_lora_loop.sh
```

### 🧪 Evaluation

Follow the same instructions as Training to specify `gpu_ids`, `hyper_1`, and `hyper_2` in `eval_loop.sh`.

```bash
cd GITA
bash ./scripts/eval/eval_loop.sh
```

For zero-shot GITA, set `EPOCH BSZ LORAR UNFREEZEV LAYOUTAUG` in `hyper_2` as `none`.

Example for zero-shot GITA-7B Vision-Only on GVLQA-BASE:

```bash
7b none none none 16 Vision_Only GVLQA-Base none none
```

## 📜 Cite Us

```bibtex
@inproceedings{wei2024gita,
  title={Gita: Graph to visual and textual integration for vision-language graph reasoning},
  author={Wei, Yanbin and Fu, Shuai and Jiang, Weisen and Zhang, Zejian and Zeng, Zhixiong and Wu, Qi and Kwok, James and Zhang, Yu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

We hope you find our repository engaging and insightful. Your journey into the realm of vision-language graph reasoning starts here! 🚀

Feel free to explore, contribute, and be a part of this exciting new venture! ✨
