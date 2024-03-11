import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import networkx as nx
from peft import PeftModel
from collections import defaultdict
from fastchat.model import get_conversation_template
import re

def apply_lora(base_model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()

    print(f"Loading the LoRA adapter from {lora_path}")
    
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16
    )
    
    print("Applying the LoRA")
    model = lora_model.merge_and_unload().cuda()
    return model, tokenizer


class Evaluation:
    def __init__(self, task):
        self.correct = {"easy": 0, "medium": 0, "hard": 0}
        self.total = {"easy": 0, "medium": 0, "hard": 0}
        self.irrelevant = {"easy": 0, "medium": 0, "hard": 0}
        self.task = task

    @staticmethod
    def is_hamiltonian_path(G, path):
        # 检查路径是否包含每个节点
        if set(path) != set(G.nodes):
            return False
        # 检查路径是否连续
        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i + 1]):
                return False
        return True

    @staticmethod
    def is_topological_order(G, order):
        # 检查路径是否包含每个节点
        if set(order) != set(G.nodes):
            return False
        # 检查路径是否满足所有有向边的顺序
        order_index = {node: i for i, node in enumerate(order)}
        for u, v in G.edges:
            if order_index[u] > order_index[v]:  # 如果u在v之后，那么这不是一个拓扑排序
                return False
        return True

    @staticmethod
    def convert_graph_path(filename):
        parts = filename.split("_dot_")
        task, remaining = parts[0].split("-", 1)
        full_flag, remaining = remaining.split("-", 1)
        level, string = remaining.split("-", 1)
        match = re.search(r'\d+', string)
        if match:
            number = match.group()
            converted_graph_path = f"/mnt/sdb1/NLGraph/NLGraph/{task}/graph/{level}/full/graph{number}.txt"
            return converted_graph_path
        else:
            raise ValueError("No found graph number!!!")

    # "cycle-full-easy-graph8_dot_ellipse_1.0_white_filled"
    # "**/topology/graph/easy/full/graph8.txt"
    def count(self, output, ground_truth, path_id):
        if self.task == "gnn":
            if output.startswith("The updated embeddings of each node:"):
                if output == ground_truth:
                    self.correct[path_id.split("-")[2]] += 1
            else:
                self.irrelevant[path_id.split("-")[2]] += 1

        elif self.task == "hamilton":
            candidate = output.split(".")[0].split('->')
            try:
                candidate = list(map(int, candidate))
                graph_path = self.convert_graph_path(path_id)

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
                    self.correct[path_id.split("-")[2]] += 1
            except ValueError:
                self.irrelevant[path_id.split("-")[2]] += 1

        elif self.task == "topology":
            candidate = output.split(".")[0].split(',')
            try:
                candidate = list(map(int, candidate))
                graph_path = self.convert_graph_path(path_id)

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
                    self.correct[path_id.split("-")[2]] += 1
            except ValueError:
                self.irrelevant[path_id.split("-")[2]] += 1

        else:
            # The answers to other tasks are in the form of xxx.
            if len(output.split(".")) == 2:
                if output == ground_truth:
                    self.correct[path_id.split("-")[2]] += 1
            else:
                self.irrelevant[path_id.split("-")[2]] += 1

        self.total[path_id.split("-")[2]] += 1
        return self.correct, self.irrelevant, self.total

@torch.inference_mode()
def get_model_answers(args):
    model_path = args.model_path
    model_id = args.model_id
    input_file = args.question_file
    output_file = args.answer_file
    
    evaluation = Evaluation(task=args.task)
    correct, irrelevant, total = None, None, None
    
    ques_jsons = []
    with open(os.path.expanduser(input_file), "r") as ques_file:
        ques_jsons = json.load(ques_file)
    
    output_file = os.path.expanduser(output_file)
    model_path = os.path.expanduser(model_path)
    
    model, tokenizer = apply_lora(
        base_model_path=model_path, lora_path=args.lora_path,
        model_max_length=args.model_max_length
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
    # ).cuda()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ans_file = open(output_file, "w")
    
    for ques in tqdm(ques_jsons):
        qa_id = ques["id"]
        qa_conv = ques['conversations']
        qs = qa_conv[0]['value']
        gt = qa_conv[1]['value']
        conv = get_conversation_template(model_id)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids

        with torch.inference_mode():
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
            )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        correct, irrelevant, total = evaluation.count(output=outputs, ground_truth=gt, path_id=qa_id)

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {"qa_id": qa_id,
                "question": qs,
                "text": outputs,
                "answer_id": ans_id,
                "ground_truth": gt}
            ) + "\n"
        )

    ans_file.write(
        json.dumps(
            {"easy accuracy": correct["easy"] / total["easy"] if total["easy"] else "null",
            "medium accuracy": correct["medium"] / total["medium"] if total["medium"] else "null",
            "hard accuracy": correct["hard"] / total["hard"] if total["hard"] else "null",
            "total": total,
            "correct": correct,
            "irrelevant": irrelevant}
        )
    )
    ans_file.close()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    get_model_answers(args)


    