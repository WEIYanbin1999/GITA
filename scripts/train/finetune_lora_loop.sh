#!/bin/bash

gpu_ids=(
    "0,1,2,3,4,5,6,7"
)

declare -a hyper_1=(
    "hamilton"
    "cycle"
    "flow"
    "matching"
    "shortest_path"
    "topology"
    "connectivity"

#    "CiteSeer"
#    "Cora"
#    "email-Eu-core"
#    "PolBlogs"

#    "ca-GrQc"
#    "ca-HepTh"
)

declare -a hyper_2=(
    "7b 5 16 64 16 Vision_Only GVLQA-AUGET True False"
    "7b 10 16 64 16 Vision_Only GVLQA-BASE True False"

    "7b 5 32 64 16 Vision_Text GVLQA-BASE False False"
    "7b 10 32 64 16 Vision_Text GVLQA-BASE False False"
)

declare -a params=()

for h1 in "${hyper_1[@]}"; do
    for h2 in "${hyper_2[@]}"; do
        params+=("${h1} ${h2}")
    done
done

for gpu_index in "${!gpu_ids[@]}"; do
    gpu_id=${gpu_ids[$gpu_index]}
    start_index=$(("$gpu_index" * ${#params[@]} / ${#gpu_ids[@]}))
    end_index=$((("$gpu_index" + 1) * ${#params[@]} / ${#gpu_ids[@]}))

    for task_index in $(seq $start_index $((end_index - 1))); do
        random_port=$(shuf -i 10000-50000 -n 1)
        bash ./scripts/train/finetune_lora.sh $gpu_id $random_port ${params[$task_index]} &
        wait $!
    done &
done

wait
echo "Training process has completed!!!"
