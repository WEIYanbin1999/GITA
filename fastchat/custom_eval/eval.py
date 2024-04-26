import argparse
import pdb

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import networkx as nx
from peft import PeftModel
from fastchat.model import get_conversation_template


def load_model(base_model_path, lora_path, test_type):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()

    if test_type == "fine-tuned":
        print(f"Loading the LoRA adapter from {lora_path}")

        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.float16
        )
    
        print("Applying the LoRA")
        model = lora_model.merge_and_unload().cuda()

    elif test_type == "zero-shot":
        model = base_model
    else:
        raise NotImplementedError("Do not support this testing type!")

    return model, tokenizer


class Evaluation:
    def __init__(self, task):
        self.correct = {"easy": 0, "medium": 0, "hard": 0}
        self.total = {"easy": 0, "medium": 0, "hard": 0}
        self.irrelevant = {"easy": 0, "medium": 0, "hard": 0}
        self.task = task

    @staticmethod
    def is_hamiltonian_path(G, path):
        # Check that the path contains each node
        if set(path) != set(G.nodes):
            return False
        # Check if the path is continuous
        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i + 1]):
                return False
        return True

    @staticmethod
    def is_topological_order(G, order):
        # Check that the path contains each node
        if set(order) != set(G.nodes):
            return False
        # Check if the path satisfies the order of all directed edges
        order_index = {node: i for i, node in enumerate(order)}
        for u, v in G.edges:
            if order_index[u] > order_index[v]:  # If u comes after v, then this is not a topological sort
                return False
        return True

    def count(self, output, ground_truth, path_id, ques_file):
        task = path_id.split("-")[0]
        task_difficulty = path_id.split("-")[1]
        graph_id = path_id.split("-")[2]
        # get graph path from ques_file and path_id
        graph_path = os.path.join("/".join(ques_file.split("/")[:4]), task,
                                  "graph_structure", task_difficulty, graph_id + ".txt")

        if self.task == "gnn":
            if output.startswith("The updated embeddings of each node:"):
                if output == ground_truth:
                    self.correct[task_difficulty] += 1
            else:
                self.irrelevant[task_difficulty] += 1

        elif self.task == "hamilton":
            candidate = output.split(".")[0].split('->')
            candidate = list(map(int, candidate))

            G = nx.Graph()
            with open(graph_path, "r") as f:
                n, m = [int(x) for x in next(f).split()]
                array = []
                for line in f:  # read rest of lines
                    array.append([int(x) for x in line.split()])
                edges = array[:m]
                assert len(edges) == m
                G.add_nodes_from(range(n))
                for edge in edges:
                    G.add_edge(edge[0], edge[1])

            if self.is_hamiltonian_path(G=G, path=candidate):
                self.correct[task_difficulty] += 1
            else:
                self.irrelevant[task_difficulty] += 1

        elif self.task == "topology":
            candidate = output.split(".")[0].split(',')
            candidate = list(map(int, candidate))

            G = nx.DiGraph()
            with open(graph_path, "r") as f:
                n, m = [int(x) for x in next(f).split()]
                array = []
                for line in f:  # read rest of lines
                    array.append([int(x) for x in line.split()])
                edges = array[:m]
                assert len(edges) == m
                G.add_nodes_from(range(n))
                for edge in edges:
                    G.add_edge(edge[0], edge[1])

            if self.is_topological_order(G=G, order=candidate):
                self.correct[task_difficulty] += 1
            else:
                self.irrelevant[task_difficulty] += 1

        else:
            # The answers to other tasks are in the form of xxx.
            if len(output.split(".")) == 2:
                if output == ground_truth:
                    self.correct[task_difficulty] += 1
            else:
                self.irrelevant[task_difficulty] += 1

        self.total[task_difficulty] += 1
        return self.correct, self.irrelevant, self.total


@torch.inference_mode()
def eval_model(args):
    base_model_path = args.base_model_path
    lora_path = args.lora_path
    answer_file = args.answer_file

    evaluation = Evaluation(task=args.task)
    correct, irrelevant, total = None, None, None

    with open(args.question_file, "r") as qf:
        ques_jsons = json.load(qf)
    
    model, tokenizer = load_model(
        base_model_path=base_model_path,
        lora_path=lora_path,
        test_type=args.test_type
    )

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    
    for ques in tqdm(ques_jsons):
        path_id = ques["id"]
        qa_conv = ques['conversations']
        qs = qa_conv[0]['value']
        gt = qa_conv[1]['value']
        conv = get_conversation_template("vicuna")
        task = path_id.split("-")[0]

        if args.test_type == "zero-shot":
            if args.task in ['cycle', 'connectivity']:
                qs += ' Note! You response should exactly contain one word: Yes. or No.'
            elif args.task in ['flow', 'matching']:
                qs += (' Note! Don\'t give me any response except directly give one number as the answer,'
                       ' for example, 3. or 8.')
            elif args.task in ['hamilton', 'shortest_path']:
                qs += (' Note! Don\'t give me any response except directly give one path as the answer,'
                       ' for example, 0->1->2->3->4. or 0->1->3->7->8->4->6->5->9->2.')
            elif args.task in ['topology']:
                qs += (' Note! Directly provide a possible topological ordering path. '
                       'No additional information or explanation is required,'
                       ' for example, 0,1,2,3,4. or 0,1,3,7,8,4,6,5,9,2.')
            else:
                raise ValueError("Do not support this task for zero-shot evaluation!")

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids

        with torch.inference_mode():
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        correct, irrelevant, total = evaluation.count(
            output=output,
            ground_truth=gt,
            path_id=path_id,
            ques_file=args.question_file
        )

        ans_file.write(
            json.dumps(
                {
                    "path_id": path_id,
                    "question": qs,
                    "output": output,
                    "ground_truth": gt
                }
            ) + "\n"
        )

    ans_file.write(
        json.dumps(
            {
                "easy accuracy": correct["easy"] / total["easy"] if total["easy"] else "null",
                "medium accuracy": correct["medium"] / total["medium"] if total["medium"] else "null",
                "hard accuracy": correct["hard"] / total["hard"] if total["hard"] else "null",
                "total": total,
                "correct": correct,
                "irrelevant": irrelevant
            }
        )
    )
    ans_file.close()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True, default="")
    parser.add_argument("--lora-path", type=str, required=True, default="")
    parser.add_argument("--question-file", type=str, required=True, default="")
    parser.add_argument('--task', type=str, required=True, default="")
    parser.add_argument('--test-type', type=str, required=True, default="", choices=["fine-tuned", "zero-shot"])
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    eval_model(args)


    