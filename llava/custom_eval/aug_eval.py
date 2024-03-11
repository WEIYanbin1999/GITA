import re
import ipdb
import argparse
import torch
import os
import json
from tqdm import tqdm
import networkx as nx

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, json_file_path, tokenizer, image_processor, model_config):
        with open(json_file_path, "r") as j:
            self.contents = json.loads(j.read())
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        image_path = self.contents[index]["image"]
        question = self.contents[index]["conversations"][0]["value"]
        image_id = self.contents[index]["id"]
        ground_truth = self.contents[index]["conversations"][1]["value"]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image_id, question, ground_truth, image_path

    def __len__(self):
        return len(self.contents)


# DataLoader
def create_data_loader(json_file_path, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(json_file_path, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


class Evaluation:
    def __init__(self, task):
        self.correct = {
            "dot": 0,
            "neato": 0,
            "circo": 0,
            "fdp": 0,
            "sfdp": 0,
            "twopi": 0,
            "ellipse": 0,
            "circle": 0,
            "box": 0,
            "polygon": 0,
            "width_1.0": 0,
            "width_2.0": 0,
            "width_4.0": 0,
            "width_8.0": 0,
            "filled": 0,
            "dashed": 0,
            "dotted": 0,
            "bold": 0,
        }
        self.total = {
            "dot": 0,
            "neato": 0,
            "circo": 0,
            "fdp": 0,
            "sfdp": 0,
            "twopi": 0,
            "ellipse": 0,
            "circle": 0,
            "box": 0,
            "polygon": 0,
            "width_1.0": 0,
            "width_2.0": 0,
            "width_4.0": 0,
            "width_8.0": 0,
            "filled": 0,
            "dashed": 0,
            "dotted": 0,
            "bold": 0,
        }
        self.irrelevant = {
            "dot": 0,
            "neato": 0,
            "circo": 0,
            "fdp": 0,
            "sfdp": 0,
            "twopi": 0,
            "ellipse": 0,
            "circle": 0,
            "box": 0,
            "polygon": 0,
            "width_1.0": 0,
            "width_2.0": 0,
            "width_4.0": 0,
            "width_8.0": 0,
            "filled": 0,
            "dashed": 0,
            "dotted": 0,
            "bold": 0,
        }
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
    def convert_graph_path(img_path, path_id):
        parts = path_id.split("_dot_")
        task, remaining = parts[0].split("-", 1)
        full_flag, remaining = remaining.split("-", 1)
        level, string = remaining.split("-", 1)
        match = re.search(r'\d+', string)
        if match:
            number = match.group()
            if "Aug" in img_path:
                converted_graph_path = f"/mnt/sdb1/Aug_Graph/NLGraph/{task}/graph/{level}/full/graph{number}.txt"
            else:
                converted_graph_path = f"/mnt/sdb1/NLGraph/NLGraph/{task}/graph/{level}/full/graph{number}.txt"
            return converted_graph_path
        else:
            raise ValueError("No found graph number!!!")

    @staticmethod
    def get_aug_name(aug_type, path_id):
        if aug_type == "6lay":
            if "dot" in path_id:
                aug_name = "dot"
                return aug_name
            elif "neato" in path_id:
                aug_name = "neato"
                return aug_name
            elif "circo" in path_id:
                aug_name = "circo"
                return aug_name
            elif "-fdp" in path_id:
                aug_name = "fdp"
                return aug_name
            elif "-sfdp" in path_id:
                aug_name = "sfdp"
                return aug_name
            elif "twopi" in path_id:
                aug_name = "twopi"
                return aug_name

        elif aug_type == "4nshape":
            if "ellipse" in path_id:
                aug_name = "ellipse"
                return aug_name
            elif "circle" in path_id:
                aug_name = "circle"
                return aug_name
            elif "box" in path_id:
                aug_name = "box"
                return aug_name
            elif "polygon" in path_id:
                aug_name = "polygon"
                return aug_name

        elif aug_type == "4ewidth":
            if "_1.0_" in path_id:
                aug_name = "width_1.0"
                return aug_name
            elif "_2.0_" in path_id:
                aug_name = "width_2.0"
                return aug_name
            elif "_4.0_" in path_id:
                aug_name = "width_4.0"
                return aug_name
            elif "_8.0_" in path_id:
                aug_name = "width_8.0"
                return aug_name

        elif aug_type == "4nstyle":
            if "filled" in path_id:
                aug_name = "filled"
                return aug_name
            elif "dashed" in path_id:
                aug_name = "dashed"
                return aug_name
            elif "dotted" in path_id:
                aug_name = "dotted"
                return aug_name
            elif "bold" in path_id:
                aug_name = "bold"
                return aug_name

        else:
            raise ValueError("No found graph number!!!")

    def count(self, aug_type, output, ground_truth, image_path, path_id):
        if self.task == "gnn":
            if output.startswith("The updated embeddings of each node:"):
                if output == ground_truth:
                    self.correct[self.get_aug_name(aug_type, path_id)] += 1
            else:
                self.irrelevant[self.get_aug_name(aug_type, path_id)] += 1

        elif self.task == "hamilton":
            candidate = output.split(".")[0].split('->')
            try:
                candidate = list(map(int, candidate))
                graph_path = self.convert_graph_path(image_path, path_id)

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
                    self.correct[self.get_aug_name(aug_type, path_id)] += 1
            except ValueError:
                self.irrelevant[self.get_aug_name(aug_type, path_id)] += 1

        elif self.task == "topology":
            candidate = output.split(".")[0].split(',')
            try:
                candidate = list(map(int, candidate))
                graph_path = self.convert_graph_path(image_path, path_id)

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
                    self.correct[self.get_aug_name(aug_type, path_id)] += 1
            except ValueError:
                self.irrelevant[self.get_aug_name(aug_type, path_id)] += 1

        else:
            if len(output.split(".")) == 2:
                if output == ground_truth:
                    self.correct[self.get_aug_name(aug_type, path_id)] += 1
            else:
                self.irrelevant[self.get_aug_name(aug_type, path_id)] += 1


        self.total[self.get_aug_name(aug_type, path_id)] += 1
        return self.correct, self.irrelevant, self.total


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    json_file_path = os.path.expanduser(args.test_json_file_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(json_file_path, tokenizer, image_processor, model.config)

    # build iterator for evaluation
    evaluation = Evaluation(task=args.task)
    correct, irrelevant, total = None, None, None
    aug_type = args.answers_file.split("/")[-2]

    for input_ids, image_tensor, path_id, question, ground_truth, image_path in tqdm(data_loader, total=len(data_loader)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if isinstance(path_id, tuple):
            path_id = path_id[0]
        if isinstance(question, tuple):
            question = question[0]
        if isinstance(ground_truth, tuple):
            ground_truth = ground_truth[0]
        if isinstance(image_path, tuple):
            image_path = image_path[0]

        with torch.inference_mode():
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
        output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output = output.strip()

        correct, irrelevant, total = evaluation.count(
            aug_type=aug_type,
            output=output,
            ground_truth=ground_truth,
            image_path=image_path,
            path_id=path_id
        )

        ans_file.write(json.dumps({"path_id": path_id,
                                   "question": question,
                                   "text": output,
                                   "ground_truth": ground_truth,
                                   "model_id": model_name}) + "\n")
        # ans_file.flush()
    ans_file.write(json.dumps({"dot accuracy": correct["dot"] / total["dot"] if total["dot"] else "null",
                               "neato accuracy": correct["neato"] / total["neato"] if total["neato"] else "null",
                               "circo accuracy": correct["circo"] / total["circo"] if total["circo"] else "null",
                               "fdp accuracy": correct["fdp"] / total["fdp"] if total["fdp"] else "null",
                               "sfdp accuracy": correct["sfdp"] / total["sfdp"] if total["sfdp"] else "null",
                               "twopi accuracy": correct["twopi"] / total["twopi"] if total["twopi"] else "null",
                               "ellipse accuracy": correct["ellipse"] / total["ellipse"] if total["ellipse"] else "null",
                               "circle accuracy": correct["circle"] / total["circle"] if total["circle"] else "null",
                               "box accuracy": correct["box"] / total["box"] if total["box"] else "null",
                               "width_1.0 accuracy": correct["width_1.0"] / total["width_1.0"] if total["width_1.0"] else "null",
                               "width_2.0 accuracy": correct["width_2.0"] / total["width_2.0"] if total["width_2.0"] else "null",
                               "width_4.0 accuracy": correct["width_4.0"] / total["width_4.0"] if total["width_4.0"] else "null",
                               "width_8.0 accuracy": correct["width_8.0"] / total["width_8.0"] if total["width_8.0"] else "null",
                               "filled accuracy": correct["filled"] / total["filled"] if total["filled"] else "null",
                               "dashed accuracy": correct["dashed"] / total["dashed"] if total["dashed"] else "null",
                               "dotted accuracy": correct["dotted"] / total["dotted"] if total["dotted"] else "null",
                               "bold accuracy": correct["bold"] / total["bold"] if total["bold"] else "null",
                               "total": total,
                               "correct": correct,
                               "irrelevant": irrelevant}))
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default="")
    parser.add_argument("--test-json-file-path", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
