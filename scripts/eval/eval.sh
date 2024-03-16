GPU_IDS=$1
TASK=$2
MODELSIZE=$3
TESTTYPE=$4  # fine-tuning or zero-shot
DATATYPE=$6  # Base, Base-Pruned, Aug, Aug-Pruned
MODALTYPE=$7  # Text_Only, Vision_Only, Vision_Text (both image and text)
EPOCH=$8  # # optional: none (when TESTTYPE=zero-shot); the number of epochs for fine-tuning (when TESTTYPE=fine-tuning)


if [ "$MODALTYPE" == "Text_Only" ]; then
    pretrained_model_path="../local_llm/vicuna-v1.5-${MODELSIZE}"
    checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"
    data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Base/${MODALTYPE}_test.json"
    answer_path="./answer/${MODALTYPE}/${DATATYPE}/vicuna-v1.5-${MODELSIZE}-${TASK}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}.answer.jsonl"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    python -m fastchat.custom_eval.eval \
        --task "$TASK" \
        --lora-path "$checkpoint_path" \
        --model-path "$pretrained_model_path" \
        --question-file "$data_path" \
        --answer-file "$answer_path" \
        --test-type "$TESTTYPE"

elif [[ "$MODALTYPE" == *"Vision"* ]]; then
    pretrained_model_path="../local_llm/llava-v1.5-${MODELSIZE}"
    image_folder="../dataset/GITQA-${DATATYPE}"
    
    if [[ ${DATATYPE} == *"Aug"* ]]; then
        # augmentation dataset path
        data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Aug/${AUGTYPE}_test.json"
        checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/${AUGTYPE}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-${EPOCH}"

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        python -m llava.custom_eval.aug_eval \
            --task "$TASK" \
            --lora-path "$checkpoint_path" \
            --model-path "$pretrained_model_path" \
            --question-file "$data_path" \
            --answer-file "$answer_path" \
            --test-type "$TESTTYPE"
    else
        # base dataset path
        data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Base/${MODALTYPE}_test.json"
        checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}--lora(${LORAR}, ${LORAALPHA})-${EPOCH}"

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        python -m llava.custom_eval.eval \
            --task "$TASK" \
            --lora-path "$checkpoint_path" \
            --model-path "$pretrained_model_path" \
            --question-file "$data_path" \
            --answer-file "$answer_path" \
            --test-type "$TESTTYPE"
    fi

else
    echo "Do not support this type of data!!!"
fi



