GPU_IDS=$1
TASK=$2
MODELSIZE=$3
EPOCH=$4
DATATYPE=$5  # full, 4ewidth, 4nshape, 4nstyle, 6lay (for text modality, only "full" available)
MODALTYPE=$6  # text, vision, vision_text (both image and text)


if [ "$DATATYPE" != "full" ]; then
    # augmentation path
    data_path="/mnt/sdb1/Aug_Graph/NLGraph/${TASK}/test_${DATATYPE}.json"
else
    data_path="/mnt/sdb1/NLGraph/NLGraph/${TASK}/test_${file_type}_${DATATYPE}.json"
fi

if [ "$MODALTYPE" == "text" ]; then
    file_type="tgqa"
    checkpoint_path="./checkpoints/text_${DATATYPE}/vicuna-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}"
    answer_path="./answer/text_${DATATYPE}/vicuna-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}-answer.jsonl"
    pretrained_model_path="/mnt/sdb1/local_llm/vicuna-v1.5-${MODELSIZE}"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    python -m fastchat.custom_eval.eval \
        --task ${TASK} \
        --lora-path $checkpoint_path \
        --model-path $pretrained_model_path \
        --model-id vicuna-${MODELSIZE}:v1 \
        --question-file $data_path \
        --answer-file $answer_path

elif [ "$MODALTYPE" == "vision" ]; then
    file_type="vgqa"
    if [ "$DATATYPE" != "full" ]; then
        # augmentation path
        checkpoint_path="./checkpoints/aug_full/${DATATYPE}/llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}"
        answer_path="./answer/aug_full/${DATATYPE}/llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}-answer.jsonl"
    else
        checkpoint_path="./checkpoints/vision_${DATATYPE}//llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}"
        answer_path="./answer/vision_${DATATYPE}/llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}-answer.jsonl"
    fi
    pretrained_model_path="/mnt/sdb1/local_llm/llava-v1.5-${MODELSIZE}"



elif [ "$MODALTYPE" == "vision_text" ]; then
    file_type="IT"
    if [ "$DATATYPE" != "full" ]; then
        # augmentation path
        checkpoint_path="./checkpoints/img_text_aug_full/${DATATYPE}/llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}"
        answer_path="./answer/img_text_aug_full/${DATATYPE}/llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}-answer.jsonl"
    else
        checkpoint_path="./checkpoints/img_text_${DATATYPE}/llava-v1.5-${MODELSIZE}-${TASK}-lora-epoch-${EPOCH}"
    fi
    pretrained_model_path="/mnt/sdb1/local_llm/llava-v1.5-${MODELSIZE}"



else
    echo "Do not support this type of data!!!"
fi



