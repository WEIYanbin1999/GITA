#!/bin/bash

gpu_ids=(
    "0"
#    "0,1,2,3,4,5,6,7"
)

#task="cycle"
#task="connectivity"
#task="flow"
#task="gnn"
#task="hamilton"
#task="matching"
#task="shortest_path"
#task="topology"

declare -a hyper_1=(
    "cycle"
#    "flow"
#    "hamilton"
#    "matching"
#    "shortest_path"
#    "topology"
#    "gnn"
#    "connectivity"
)

declare -a hyper_2=(
    "7b 5 16 Text_Only Base-Pruned none"
#    "7b 10 16"
#    "7b 20 16"
#    "7b 30 16"
#    "13b 5 8"
#    "13b 10 8"
#    "13b 20 8"
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
            bash ./scripts/train/finetune_lora.sh "$gpu_id" "$random_port" "${params[$task_index]}" &
            wait $!
      done &
done

wait
python send_email "116: Training process has completed"

