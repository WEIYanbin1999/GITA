import os
import torch
import graphviz
import random
import json
import numpy as np
from copy import deepcopy
from torch.backends import cudnn
from torch_geometric.utils import k_hop_subgraph, negative_sampling
from tqdm import tqdm


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    seed_torch(worker_seed % (2**32))


def create_data_object(data, pos_edge_index, neg_edge_index):
    new_data = deepcopy(data)
    new_data.edge_index = pos_edge_index
    new_data.neg_edge_index = neg_edge_index
    return new_data


def add_mask(test_or_val_edges, train_edge_index, train_neg_edge_index):
    for edge in test_or_val_edges.t():
        edge = edge.unsqueeze(1)
        mask = ~torch.all(train_edge_index == edge, dim=0)
        train_edge_index = train_edge_index[:, mask]
        train_neg_edge_index = train_neg_edge_index[:, mask]
    return train_edge_index, train_neg_edge_index


def link_prediction_custom_split(data, train_ratio=0.8, val_ratio=0.1, split_case='train_val_test'):
    num_edges = data.edge_index.size(1)
    num_train_edges = int(num_edges * train_ratio)
    num_val_edges = int(num_edges * val_ratio)
    num_test_edges = num_edges - num_train_edges - num_val_edges

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_neg_samples=num_edges,
        force_undirected=True,
    )

    pos_perm = torch.randperm(num_edges)
    if split_case == 'train_val_test':
        train_edge_index = data.edge_index[:, pos_perm[:num_train_edges]]
        val_edge_index = data.edge_index[:, pos_perm[num_train_edges:num_train_edges + num_val_edges]]
        test_edge_index = data.edge_index[:, pos_perm[num_train_edges + num_val_edges:]]
    elif split_case == 'train_test_val':
        train_edge_index = data.edge_index[:, pos_perm[:num_train_edges]]
        test_edge_index = data.edge_index[:, pos_perm[num_train_edges:num_train_edges + num_test_edges]]
        val_edge_index = data.edge_index[:, pos_perm[num_train_edges + num_test_edges:]]
    else:
        raise NotImplementedError("Do not support this split method.")

    neg_perm = torch.randperm(num_edges)
    if split_case == 'train_val_test':
        train_neg_edge_index = neg_edge_index[:, neg_perm[:num_train_edges]]
        val_neg_edge_index = neg_edge_index[:, neg_perm[num_train_edges:num_train_edges + num_val_edges]]
        test_neg_edge_index = neg_edge_index[:, neg_perm[num_train_edges + num_val_edges:]]
    elif split_case == 'train_test_val':
        train_neg_edge_index = neg_edge_index[:, neg_perm[:num_train_edges]]
        test_neg_edge_index = neg_edge_index[:, neg_perm[num_train_edges:num_train_edges + num_test_edges]]
        val_neg_edge_index = neg_edge_index[:, neg_perm[num_train_edges + num_test_edges:]]
    else:
        raise NotImplementedError("Do not support this split method.")

    test_edges = torch.cat([test_edge_index, torch.flip(test_edge_index, dims=[0])], dim=1)
    val_edges = torch.cat([val_edge_index, torch.flip(val_edge_index, dims=[0])], dim=1)
    train_edge_index, train_neg_edge_index = add_mask(test_edges, train_edge_index, train_neg_edge_index)
    train_edge_index, train_neg_edge_index = add_mask(val_edges, train_edge_index, train_neg_edge_index)

    train_data = create_data_object(data, train_edge_index, train_neg_edge_index)
    val_data = create_data_object(data, val_edge_index, val_neg_edge_index)
    test_data = create_data_object(data, test_edge_index, test_neg_edge_index)

    return train_data, val_data, test_data


class MyCaArxivDataset(object):
    def __init__(self, task_name, save_path="./"):
        num_node_dict = {'ca-GrQc': 5242, 'ca-HepTh': 9877}
        self.task_name = task_name
        self.save_path = save_path
        self.num_nodes = num_node_dict[task_name]
        self.data = type('Data', (object,), {})()
        self.data.edge_index = torch.tensor(self.read_edges(), dtype=torch.long).t().contiguous()

    def read_edges(self):
        edge_list = []
        with open(os.path.join(self.save_path, f"dataset/{self.task_name}/{self.task_name}.txt"), "r") as file:
            for line in file:
                node1, node2 = map(int, line.split())
                node1, node2 = (min(node1, node2), max(node1, node2))
                if [node1, node2] not in edge_list:
                    edge_list.append([node1, node2])
        return edge_list

    def get_split(self, train_split_ratio=0.8, val_split_ratio=0.1):
        return link_prediction_custom_split(self.data, train_split_ratio, val_split_ratio)


class GraphVisualizer:
    def __init__(self):
        self.layout_list = ['dot', 'neato', 'circo', 'twopi', 'fdp', 'sfdp']
        
    def convert_graph_to_image(self, src_node_index, dst_node_index, edge_index, file_path,
                               store_flag='vary', layout_aug=False):
        src_node_index, dst_node_index = (min(src_node_index, dst_node_index), max(src_node_index, dst_node_index))
        if store_flag == 'vary' or not os.path.exists(file_path):
            file_path = file_path.split('.png')[0]
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            
            if layout_aug:
                dot = graphviz.Graph(format='png', engine=random.choice(self.layout_list))
            else:
                dot = graphviz.Graph(format='png', engine='sfdp')

            unique_nodes = torch.unique(edge_index).cpu().numpy()

            for node in unique_nodes:
                if node not in [src_node_index, dst_node_index]:
                    dot.node(str(node), shape='box')

            for node in [src_node_index, dst_node_index]:
                dot.node(str(node), shape='box', color='brown', style='filled')
            # Add edges, using subgraph labels
            edge_index = edge_index.t().tolist()
            edge_list = []
            for start, end in edge_index:
                start, end = (min(start, end), max(start, end))
                if (start, end) != (src_node_index, dst_node_index):
                    if (start, end) not in edge_list:
                        dot.edge(str(start), str(end))
                        edge_list.append((start, end))

            dot.render(filename=file_path, cleanup=True)


class GraphDescriber:
    def __init__(self):
        self.graph_template = ('In an undirected graph, (i,j) means that node i and node j are connected with an'
                               ' undirected edge. The nodes are numbered from [P] to [P], and the edges are:\n')
        self.edge_template = '([P],[P])\n'

    def convert_graph_to_description(self, src_node_index, dst_node_index, edge_index, num_nodes):
        src_node_index, dst_node_index = (min(src_node_index, dst_node_index), max(src_node_index, dst_node_index))
        edge_list = []
        graph_description = self.graph_template.replace('[P]', '0', 1).replace('[P]', str(num_nodes-1), 1)
        edge_index = edge_index.t().tolist()
        
        num_effective_edge = 0
        for start, end in edge_index:
            start, end = (min(start, end), max(start, end))
            if (start, end) != (src_node_index, dst_node_index):
                if (start, end) not in edge_list:
                    graph_description += self.edge_template.replace('[P]', str(start), 1).replace('[P]', str(end), 1)
                    edge_list.append((start, end))
                    num_effective_edge += 1
        if num_effective_edge == 0:
            graph_description += 'None.\n'
        return graph_description


class Questioner:
    def __init__(self):
        self.visual_graph_template = ("<image>\nThis image depicts an undirected graph composed of rectangle nodes"
                                      " labeled with numbers, and edges between nodes.\n")
        self.visual_text_transition = "The following paragraph describes the same graph as the image: "
        self.task_responsibility = ("The task is link prediction, aiming to predict the presence or absence of an"
                                    " unknown edge between Node [P] and Node [P] based on the known graph structure. ")
        self.output_specification = "Q: Does an unknown edge exist between Node [P] and Node [P]?"
        
    def generate_vt_query(self, graph_description, src_node_index, dst_node_index):
        src_node_index, dst_node_index = (min(src_node_index, dst_node_index), max(src_node_index, dst_node_index))
        query = ""
        query += self.visual_graph_template
        query += self.visual_text_transition
        query += graph_description
        query += self.task_responsibility.replace('[P]', str(src_node_index), 1).replace('[P]', str(dst_node_index), 1)
        query += self.output_specification.replace('[P]', str(src_node_index), 1).replace('[P]', str(dst_node_index), 1)
        return query

    def generate_t_query(self, graph_description, src_node_index, dst_node_index):
        src_node_index, dst_node_index = (min(src_node_index, dst_node_index), max(src_node_index, dst_node_index))
        query = ""
        query += graph_description
        query += self.task_responsibility.replace('[P]', str(src_node_index), 1).replace('[P]', str(dst_node_index), 1)
        query += self.output_specification.replace('[P]', str(src_node_index), 1).replace('[P]', str(dst_node_index), 1)
        return query
    
    
class DataConstructor:
    def __init__(self, task_name, modalities='Vision_Text', save_path="./", layout_aug=False):
        self.layout_aug = layout_aug
        self.modalities = modalities
        self.task_name = task_name
        self.save_path = save_path
        if task_name in ['ca-GrQc', 'ca-HepTh']:
            my_data = MyCaArxivDataset(task_name, save_path)
        else:
            raise NotImplementedError("Do not support this task.")
            
        # 8: 1: 1 split
        self.train_data, self.val_data, self.test_data = my_data.get_split(train_split_ratio=0.8, val_split_ratio=0.1)
        self.num_nodes = my_data.num_nodes
        
        # Link Prediction Structure Setting
        self.train_val_data = deepcopy(self.train_data)
        self.train_val_data.edge_index = torch.cat([self.train_data.edge_index, self.val_data.edge_index], dim=1)
        self.train_val_data.neg_edge_index = torch.cat([self.train_data.neg_edge_index,
                                                        self.val_data.neg_edge_index], dim=1)
        train_test_data = deepcopy(self.train_data)
        train_test_data.edge_index = torch.cat([self.train_data.edge_index, self.test_data.edge_index], dim=1)
        train_test_data.neg_edge_index = torch.cat([self.train_data.neg_edge_index,
                                                    self.test_data.neg_edge_index], dim=1)
        
        train_link_tensor, val_link_tensor, test_link_tensor = (self.train_data.edge_index.t(),
                                                                self.val_data.edge_index.t(),
                                                                self.test_data.edge_index.t())
        test_nolink_tensor = self.test_data.neg_edge_index.t()
        self.train_pos_edges = [tuple(row.tolist()) for row in train_link_tensor]
        self.test_pos_edges = [tuple(row.tolist()) for row in test_link_tensor]
        self.val_pos_edges = [tuple(row.tolist()) for row in val_link_tensor]
        self.test_edges = self.test_pos_edges + [tuple(row.tolist()) for row in test_nolink_tensor]
               
        # GITA components
        self.graph_visualizer = GraphVisualizer()
        self.graph_describer = GraphDescriber()
        self.questioner = Questioner()
        
    def construct_json(self, data_split='train'):
        # Prepare link prediction setting structures
        all_samples = []

        if data_split == 'train':
            selected_pos_edges = self.train_pos_edges
            selected_neg_edges = []
        
            while len(selected_neg_edges) < len(selected_pos_edges):
                random_nega = (random.randint(0, self.num_nodes-1), random.randint(0, self.num_nodes-1))
                if random_nega in selected_pos_edges:
                    continue
                if random_nega in selected_neg_edges or random_nega[0] == random_nega[1]:
                    continue
                if random_nega in self.test_pos_edges or random_nega in self.val_pos_edges:
                    continue
                selected_neg_edges.append(random_nega)
            all_edges = selected_pos_edges + selected_neg_edges
        elif data_split == 'test':
            selected_pos_edges = self.test_pos_edges
            all_edges = self.test_edges
        else:
            raise NotImplementedError("Do not support this split method.")
        
        random.shuffle(all_edges)
        for center_edge in tqdm(all_edges):
            sample = {}
            pos_edge_flag = True if center_edge in selected_pos_edges else False
            answer = 'Yes.' if pos_edge_flag else 'No.'
            
            src_node, dst_node = center_edge
            
            visible_graph = self.train_data if data_split == 'train' else self.train_val_data 
            tmp_visible_graph = deepcopy(visible_graph)
            
            if not pos_edge_flag: 
                new_edge = torch.tensor([[src_node], [dst_node]], dtype=torch.long)
                tmp_visible_graph.edge_index = torch.cat([visible_graph.edge_index, new_edge], dim=1)
            
            edge_index = tmp_visible_graph.edge_index
            reverse_edge_index = edge_index[[1, 0], :]
            tmp_visible_graph.edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
            
            subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
                [src_node, dst_node], num_hops=1, edge_index=tmp_visible_graph.edge_index, relabel_nodes=True
            )
            # Get the node index in the subgraph
            subgraph_node_indices = subgraph_nodes.numpy()

            # Find the new index of the original "src node" and "dst node" in the subgraph
            src_node_subgraph_index = (subgraph_node_indices == src_node).nonzero()[0].item()
            dst_node_subgraph_index = (subgraph_node_indices == dst_node).nonzero()[0].item()
            
            mask = ((subgraph_edge_index[0] == src_node_subgraph_index) &
                    (subgraph_edge_index[1] == dst_node_subgraph_index))
            subgraph_edge_index = subgraph_edge_index[:, ~mask]
            graph_description = self.graph_describer.convert_graph_to_description(
                src_node_subgraph_index, dst_node_subgraph_index,
                subgraph_edge_index, len(subgraph_nodes)
            )
            
            query = ""
            if self.modalities == 'Vision_Text':
                image_path = f"data/{self.task_name}/image/{data_split}/subgraph_image_{src_node}_{dst_node}.png"
                self.graph_visualizer.convert_graph_to_image(
                    src_node_subgraph_index, dst_node_subgraph_index, subgraph_edge_index,
                    os.path.join(self.save_path, image_path),
                    'store', layout_aug=self.layout_aug
                )
                query = self.questioner.generate_vt_query(graph_description, src_node_subgraph_index,
                                                          dst_node_subgraph_index)
                sample['image'] = image_path
            elif self.modalities == 'Text_Only':
                query = self.questioner.generate_t_query(graph_description, src_node_subgraph_index,
                                                         dst_node_subgraph_index)

            sample['id'] = self.task_name + '-' + str(src_node) + '_' + str(dst_node)
            sample["conversations"] = []
            sample["conversations"].append({
                "from": "human",
                "value": query
            })
            sample["conversations"].append({
                "from": "gpt",
                "value": answer
            })
            all_samples.append(sample)
        file_path = os.path.join(self.save_path, f"data/{self.task_name}/{self.modalities}_{data_split}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(all_samples, f, indent=4)


if __name__ == "__main__":
    # Initialization: Generate all the data once
    # Use: Call data_constructor.construct_json() for data construction before each epoch
    seed_torch()
    data_constructor = DataConstructor(task_name='ca-GrQc',
                                       modalities='Text_Only',
                                       save_path="../dataset/LINKPRED",
                                       layout_aug=False)
    data_constructor.construct_json(data_split="train")
    data_constructor.construct_json(data_split="test")

    data_constructor = DataConstructor(task_name='ca-GrQc',
                                       modalities='Vision_Text',
                                       save_path="../dataset/LINKPRED",
                                       layout_aug=False)
    data_constructor.construct_json(data_split="train")
    data_constructor.construct_json(data_split="test")

    data_constructor = DataConstructor(task_name='ca-HepTh',
                                       modalities='Text_Only',
                                       save_path="../dataset/LINKPRED",
                                       layout_aug=False)
    data_constructor.construct_json(data_split="train")
    data_constructor.construct_json(data_split="test")

    data_constructor = DataConstructor(task_name='ca-HepTh',
                                       modalities='Vision_Text',
                                       save_path="../dataset/LINKPRED",
                                       layout_aug=False)
    data_constructor.construct_json(data_split="train")
    data_constructor.construct_json(data_split="test")
