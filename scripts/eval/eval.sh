GPU_IDS=$1
TASK=$2
MODELSIZE=$3
EPOCH=$4  # optional: none (when TESTTYPE=zero-shot); the number of fine-tuned epochs (when TESTTYPE=fine-tuned)
LORAR=$5  # optional: none; the rank of the low-rank matrices used in the LoRA adaptation (default: 128)
LORAALPHA=$6  # optional: none; the scaling factor that controls the magnitude of the low-rank adaptation (default: 256)
TESTTYPE=$7  # fine-tuned or zero-shot, default: fine-tuned
DATATYPE=$8  # Base, Base-Pruned, Aug, Aug-Pruned
MODALTYPE=$9  # Text_Only, Vision_Only, Vision_Text (both image and text)


if [ "$MODALTYPE" == "Text_Only" ]; then
    pretrained_model_path="../local_llm/vicuna-v1.5-${MODELSIZE}"
    checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"
    data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Base/${MODALTYPE}_test.json"
    if [ "$TESTTYPE" == "fine-tuned" ]; then
        answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}-answer.jsonl"
    else
        answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-zero-shot-answer.jsonl"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    python -m fastchat.custom_eval.eval \
        --task "$TASK" \
        --lora-path "$checkpoint_path" \
        --base-model-path "$pretrained_model_path" \
        --question-file "$data_path" \
        --answer-file "$answer_path" \
        --test-type "$TESTTYPE"

elif [[ "$MODALTYPE" == *"Vision"* ]]; then
    pretrained_model_path="../local_llm/llava-v1.5-${MODELSIZE}"
    image_folder="../dataset/GITQA-${DATATYPE}"
    
    if [[ ${DATATYPE} == *"Aug"* ]]; then
        # augmentation dataset path
        data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Aug/${AUGTYPE}_test.json"
        if [ "$TESTTYPE" == "fine-tuned" ]; then
            answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/${AUGTYPE}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}-answer.jsonl"
            checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/${AUGTYPE}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"
        else
            answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/${AUGTYPE}/llava-v1.5-${MODELSIZE}-zero-shot-answer.jsonl"
            checkpoint_path="$pretrained_model_path"
        fi

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        python -m llava.custom_eval.eval \
            --task "$TASK" \
            --lora-path "$checkpoint_path" \
            --model-path "$pretrained_model_path" \
            --question-file "$data_path" \
            --answer-file "$answer_path" \
            --test-type "$TESTTYPE"
    else
        # base dataset path
        data_path="../dataset/GITQA-${DATATYPE}/data/${TASK}/QA/Base/${MODALTYPE}_test.json"
        if [ "$TESTTYPE" == "fine-tuned" ]; then
            answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}-answer.jsonl"
            checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"
        else
            answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-zero-shot-answer.jsonl"
            checkpoint_path="$pretrained_model_path"
        fi

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



