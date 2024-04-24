#!/bin/bash

GPU_IDS=$1
PORT=$2
TASK=$3  # GITQA: cycle, connectivity, ...; NODECLS: CiteSeer, Cora, email-Eu-core, PolBlogs; LINKPRED: ca-GrQc, ca-HepTh
MODELSIZE=$4
EPOCH=$5
BSZ=$6
LORAR=$7  # The rank of the low-rank matrices used in the LoRA adaptation (default: 64)
LORAALPHA=$8  # The scaling factor that controls the magnitude of the low-rank adaptation (default: 16)
MODALTYPE=$9  # Text_Only, Vision_Only, Vision_Text (both image and text)
DATATYPE=${10}  # GITQA-BASE, GITQA-AUGET, GITQA-AUGLY, GITQA-AUGNO, GITQA-AUGNS; NODECLS; LINKPRED
UNFREEZEV=${11}  # Optional: Fine-tune vision tower or not when Vision_Only or Vision_Text. If True, yes. (default: True)


wandb offline
gradient_accumulation_steps=$((128 / "$BSZ"))
if [[ "$DATATYPE" == *"GITQA"* ]]; then
    data_path="../dataset/${DATATYPE}/data/${TASK}/QA/${MODALTYPE}_train.json"
else
    # i.e. NODECLS or LINKPRED
    data_path="../dataset/${DATATYPE}/data/${TASK}/${MODALTYPE}_train.json"
fi

if [ "$MODALTYPE" == "Text_Only" ]; then
    pretrained_model_path="../local_llm/vicuna-v1.5-${MODELSIZE}"
    checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"

    if [ -f "$checkpoint_path/adapter_config.json" ]; then
        echo "Checkpoint file already exist!!!"
    else
        deepspeed --include localhost:"$GPU_IDS" --master_port "$PORT" fastchat/train/train_lora.py \
            --model_name_or_path "$pretrained_model_path" \
            --lora_r "$LORAR" --lora_alpha "$LORAALPHA" --lora_dropout 0.05 \
            --learning_rate 2e-4 \
            --data_path "$data_path" \
            --output_dir "$checkpoint_path" \
            --num_train_epochs "$EPOCH" \
            --fp16 True \
            --per_device_train_batch_size "$BSZ" \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps "$gradient_accumulation_steps" \
            --evaluation_strategy "no" \
            --eval_steps 100  \
            --save_strategy "steps" \
            --save_steps 500000 \
            --save_total_limit 1 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_strategy "steps" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 4096 \
            --q_lora False \
            --deepspeed ./scripts/zero3_offload.json \
            --gradient_checkpointing True \
            --flash_attn False
    fi

elif [[ "$MODALTYPE" == "Vision_Only" || "$MODALTYPE" == "Vision_Text" ]]; then
    pretrained_model_path="../local_llm/llava-v1.5-${MODELSIZE}"
    image_folder="../dataset/${DATATYPE}"

    checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-unfreeze_vit-${UNFREEZEV}-epoch-${EPOCH}"

    if [ -f "$checkpoint_path/adapter_config.json" ]; then
        echo "Checkpoint file already exist!!!"
    else
        deepspeed --include localhost:"$GPU_IDS" --master_port "$PORT" llava/train/train_mem.py \
            --lora_enable True --lora_r "$LORAR" --lora_alpha "$LORAALPHA" \
            --learning_rate 2e-4 \
            --deepspeed ./scripts/zero3_offload.json \
            --model_name_or_path "$pretrained_model_path" \
            --version v1 \
            --image_folder "$image_folder" \
            --data_path "$data_path" \
            --vision_tower openai/clip-vit-large-patch14-336 \
            --unfreeze_mm_vision_tower "$UNFREEZEV" \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length False \
            --bf16 True \
            --output_dir "$checkpoint_path" \
            --num_train_epochs "$EPOCH" \
            --per_device_train_batch_size "$BSZ" \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps "$gradient_accumulation_steps" \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 500000 \
            --save_total_limit 1 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 4096 \
            --gradient_checkpointing True \
            --dataloader_num_workers 8 \
            --lazy_preprocess True \
            --report_to wandb
    fi

else
    echo "Do not support this type of data!!!"
fi




