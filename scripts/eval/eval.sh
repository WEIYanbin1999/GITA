GPU_IDS=$1
TASK=$2  # GITQA: cycle, connectivity, ...; NODECLS: CiteSeer, Cora, email-Eu-core, PolBlogs; LINKPRED: ca-GrQc, ca-HepTh
MODELSIZE=$3
EPOCH=$4  # optional: none (when TESTTYPE=zero-shot); the number of fine-tuned epochs (when TESTTYPE=fine-tuned)
LORAR=$5  # optional: none; the rank of the low-rank matrices used in the LoRA adaptation (default: 64)
LORAALPHA=$6  # optional: none; the scaling factor that controls the magnitude of the low-rank adaptation (default: 16)
TESTTYPE=$7  # fine-tuned or zero-shot, default: fine-tuned
MODALTYPE=$8  # Text_Only, Vision_Only, Vision_Text (both image and text)
DATATYPE=$9  # GITQA-BASE, GITQA-AUGET, GITQA-AUGLY, GITQA-AUGNO, GITQA-AUGNS; NODECLS; LINKPRED
UNFREEZEV=${10}  # Optional: Fine-tune vision tower or not when Vision_Only or Vision_Text. If True, yes. (default: True)
LAYOUTAUG=${11}  # Optional: Execute layout augmentation when training large graph data or not when Vision_Only or Vision_Text. (default: True)


if [[ "$DATATYPE" == *"GITQA"* ]]; then
    data_path="../dataset/${DATATYPE}/data/${TASK}/QA/${MODALTYPE}_test.json"
else
    # i.e. NODECLS or LINKPRED
    data_path="../dataset/${DATATYPE}/data/${TASK}/${MODALTYPE}_test.json"
fi

if [ "$MODALTYPE" == "Text_Only" ]; then
    pretrained_model_path="../local_llm/vicuna-v1.5-${MODELSIZE}"
    checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/vicuna-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-epoch-${EPOCH}"

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

    if [ "$TESTTYPE" == "fine-tuned" ]; then
        answer_path="./answer/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-unfreeze_vit-${UNFREEZEV}-layout_aug-${LAYOUTAUG}-epoch-${EPOCH}-answer.jsonl"
        checkpoint_path="./checkpoints/${MODALTYPE}/${DATATYPE}/${TASK}/llava-v1.5-${MODELSIZE}-lora(${LORAR}, ${LORAALPHA})-unfreeze_vit-${UNFREEZEV}-layout_aug-${LAYOUTAUG}-epoch-${EPOCH}"
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

else
    echo "Do not support this type of data!!!"
fi



