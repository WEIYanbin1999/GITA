import argparse
import pdb

import torch
import os
import json
from tqdm import tqdm
import networkx as nx

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, args, question_file, tokenizer, image_processor, model_config, test_type, task):
        with open(question_file, "r") as j:
            self.contents = json.loads(j.read())
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.test_type = test_type
        self.task = task
        self.question_file = question_file
        self.args = args

    def __getitem__(self, index):
        path_id = self.contents[index]["id"]
        image_path = os.path.join("/".join(self.question_file.split("/")[:3]), self.contents[index]["image"])
        qs = self.contents[index]["conversations"][0]["value"]
        gt = self.contents[index]["conversations"][1]["value"]

        if self.test_type == "zero-shot":
            if self.task in ['cycle', 'connectivity']:
                qs += ' Note! You response should exactly contain one word: Yes. or No.'
            elif self.task in ['flow', 'matching']:
                qs += (' Note! Don\'t give me any response except directly give one number as the answer,'
                       ' for example, 3. or 8.')
            elif self.task in ['hamilton', 'shortest_path']:
                qs += (' Note! Don\'t give me any response except directly give one path as the answer,'
                       ' for example, 0->1->2->3->4. or 0->1->3->7->8->4->6->5->9->2.')
            elif self.task in ['topology']:
                qs += (' Note! Directly provide a possible topological ordering path. '
                       'No additional information or explanation is required,'
                       ' for example, 0,1,2,3,4. or 0,1,3,7,8,4,6,5,9,2.')
            else:
                raise ValueError("Do not support this task for zero-shot evaluation!")

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, path_id, qs, gt

    def __len__(self):
        return len(self.contents)


# DataLoader
def create_data_loader(args, question_file, tokenizer, image_processor, model_config, test_type, task,
                       batch_size=1, num_workers=2):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(args, question_file, tokenizer, image_processor, model_config, test_type, task)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


class Evaluation:
    def __init__(self, task):
        self.correct = {"easy": 0, "medium": 0, "hard": 0, "average": 0}
        self.total = {"easy": 0, "medium": 0, "hard": 0, "average": 0}
        self.irrelevant = {"easy": 0, "medium": 0, "hard": 0, "average": 0}
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

    def count(self, output, ground_truth, path_id, ques_file_path):
        # task = path_id.split("-")[0]
        if self.task not in {"CiteSeer", "Cora", "email-Eu-core", "PolBlogs", "ca-GrQc", "ca-HepTh"}:
            task_difficulty = path_id.split("-")[1]
            graph_id = path_id.split("-")[2]
            # get graph path from ques_file_path and path_id
            graph_path = os.path.join("/".join(ques_file_path.split("/")[:4]), self.task,
                                      "graph_structure", task_difficulty, graph_id + ".txt")
        else:
            task_difficulty = None

        if self.task == "hamilton":
            candidate = output.split(".")[-1].split(":")[-1].split("->")
            if not candidate:
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
                    self.correct["average"] += 1
                else:
                    self.irrelevant[task_difficulty] += 1
                    self.irrelevant["average"] += 1
            else:
                self.irrelevant[task_difficulty] += 1
                self.irrelevant["average"] += 1

        elif self.task == "topology":
            candidate = output.split(".")[0].split(',')

            if not candidate:
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
                    self.correct["average"] += 1
                else:
                    self.irrelevant[task_difficulty] += 1
                    self.irrelevant["average"] += 1
            else:
                self.irrelevant[task_difficulty] += 1
                self.irrelevant["average"] += 1

        else:
            # The answers to other tasks are in the form of xxx.
            if len(output.split(".")) == 2 or (output.isdigit() and ground_truth.isdigit()):
                if output == ground_truth:
                    if task_difficulty is not None:
                        self.correct[task_difficulty] += 1
                    self.correct["average"] += 1
            else:
                if task_difficulty is not None:
                    self.irrelevant[task_difficulty] += 1
                self.irrelevant["average"] += 1

        if task_difficulty is not None:
            self.total[task_difficulty] += 1
        self.total["average"] += 1
        return self.correct, self.irrelevant, self.total


@torch.inference_mode()
def eval_model(args):
    lora_path = args.lora_path
    question_file = args.question_file
    answer_file = args.answer_file

    model_name = get_model_name_from_path(lora_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(lora_path, args.model_path, model_name)

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    data_loader = create_data_loader(args, question_file, tokenizer, image_processor, model.config,
                                     args.test_type, args.task)

    # build iterator for evaluation
    evaluation = Evaluation(task=args.task)
    correct, irrelevant, total = None, None, None

    for input_ids, image_tensor, path_id, qs, gt in tqdm(data_loader, total=len(data_loader)):
        input_ids = input_ids[:, :4050]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if isinstance(path_id, tuple):
            path_id = path_id[0]
        if isinstance(qs, tuple):
            qs = qs[0]
        if isinstance(gt, tuple):
            gt = gt[0]

        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        correct, irrelevant, total = evaluation.count(
            output=output,
            ground_truth=gt,
            path_id=path_id,
            ques_file_path=args.question_file
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
                "average accuracy": correct["average"] / total["average"],
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
    parser.add_argument("--task", type=str, default="")
    parser.add_argument('--test-type', type=str, required=True, default="", choices=["fine-tuned", "zero-shot"])
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    arg = parser.parse_args()

    eval_model(args=arg)
