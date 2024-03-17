#!/bin/bash

GPU_IDS=$1
PORT=$2
TASK=$3
MODELSIZE=$4
EPOCH=$5
BSZ=$6
LORAR=$7  # the rank of the low-rank matrices used in the LoRA adaptation (default: 128)
LORAALPHA=$8  # the scaling factor that controls the magnitude of the low-rank adaptation (default: 256)
MODALTYPE=$9  # Text_Only, Vision_Only, Vision_Text (both image and text)
DATATYPE=${10}  # Base, Base-Pruned, Aug, Aug-Pruned
AUGTYPE=${11}  # optional: none (when DATATYPE=Base or Base-Pruned); edge_thickness, layout, node_shape, node_style (when DATATYPE=Aug or Aug-Pruned)


wandb offline
if [ "$MODALTYPE" == "Text_Only" ]; then
    pretrained_model_path="../local_llm/vicuna-v1.5-${MODELSIZE}"
    checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"
    data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Base/${MODALTYPE}_train.json"

    if [ -f "$checkpoint_path/adapter_config.json" ]; then
        echo "Checkpoint file already exist!!!"
    else
        deepspeed --include localhost:"$GPU_IDS" --master_port "$PORT" fastchat/train/train_lora.py \
            --model_name_or_path "$pretrained_model_path" \
            --lora_r "$LORAR" --lora_alpha "$LORAALPHA" --lora_dropout 0.05 \
            --data_path "$data_path" \
            --output_dir "$checkpoint_path" \
            --num_train_epochs "$EPOCH" \
            --fp16 True \
            --per_device_train_batch_size "$BSZ" \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --evaluation_strategy "no" \
            --eval_steps 100  \
            --save_strategy "steps" \
            --save_steps 500000 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_strategy "steps" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 4096 \
            --q_lora False \
            --deepspeed ./scripts/zero3.json \
            --gradient_checkpointing True \
            --flash_attn False
    fi

elif [[ "$MODALTYPE" == *"Vision"* ]]; then
    pretrained_model_path="../local_llm/llava-v1.5-${MODELSIZE}"
    image_folder="../dataset/GITQA-${DATATYPE}"

    if [[ ${DATATYPE} == *"Aug"* ]]; then
        # augmentation dataset path
        data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Aug/${AUGTYPE}_train.json"
        checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/${AUGTYPE}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-${EPOCH}"
    else
        # base dataset path
        data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Base/${MODALTYPE}_train.json"
        checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-${EPOCH}"
    fi

    if [ -f "$checkpoint_path/adapter_config.json" ]; then
        echo "Checkpoint file already exist!!!"
    else
        deepspeed --include localhost:"$GPU_IDS" --master_port "$PORT" llava/train/train_mem.py \
            --lora_enable True --lora_r "$LORAR" --lora_alpha "$LORAALPHA" \
            --mm_projector_lr 2e-5 \
            --deepspeed ./scripts/zero3.json \
            --model_name_or_path "$pretrained_model_path" \
            --version v1 \
            --image_folder "$image_folder" \
            --data_path "$data_path" \
            --vision_tower openai/clip-vit-large-patch14-336 \
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
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 500000 \
            --save_total_limit 1 \
            --learning_rate 2e-4 \
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




