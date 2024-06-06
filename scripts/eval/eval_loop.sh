#!/bin/bash

gpu_ids=(
    "6"
)

declare -a hyper_1=(
     "cycle"
     "flow"
     "hamilton"
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
    "7b none none none zero-shot Text_Only GVLQA-BASE none none"
    "13b none none none zero-shot Text_Only GVLQA-BASE none none"

#    "7b none none none zero-shot Vision_Text GVLQA-BASE none none"
#    "13b none none none zero-shot Vision_Text GVLQA-BASE none none"

#    "7b none none none zero-shot Vision_Only GVLQA-BASE none none"
#    "13b none none none zero-shot Vision_Only GVLQA-BASE none none"

#    "7b 5 64 16 fine-tuned Vision_Text NODECLS True False"
#    "7b 10 64 16 fine-tuned Vision_Text NODECLS True False"
#    "7b 5 64 16 fine-tuned Vision_Text NODECLS True True"
#    "7b 10 64 16 fine-tuned Vision_Text NODECLS True True"
#    "7b 5 64 16 fine-tuned Vision_Text NODECLS False False"
#    "7b 10 64 16 fine-tuned Vision_Text NODECLS False False"
#    "7b 5 64 16 fine-tuned Vision_Text NODECLS False True"
#    "7b 10 64 16 fine-tuned Vision_Text NODECLS False True"
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
            bash ./scripts/eval/eval.sh $gpu_id ${params[$task_index]} &
            wait $!
      done &
done

wait
echo "Evaluation process has completed!!!"
