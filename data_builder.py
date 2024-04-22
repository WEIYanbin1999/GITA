import os
import torch
import graphviz
import random
import numpy as np
from torch_geometric.loader import NeighborSampler
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.backends import cudnn
from data_modules import MyPolblogsDataset, MyPlanetoidDataset, MyEmailEuCoreDataset


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32) + worker_id
    seed_torch(worker_seed % (2 ** 32))


class GraphVisualizer(object):
    def __init__(self, num_class, data_name, save_path="./"):
        self.num_class = num_class
        self.light_colors = [self.random_light_color() for _ in range(num_class)]
        self.deep_colors = [self.random_deep_color() for _ in range(num_class)]
        self.layout_list = ["dot", "neato", "circo", "twopi", "fdp", "sfdp"]
        self.visualize_colors(data_name=data_name, save_path=save_path)

    @staticmethod
    def random_light_color():
        r = random.randint(175, 255)
        g = random.randint(150, 255)
        b = random.randint(160, 255)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    @staticmethod
    def random_deep_color():
        r = random.randint(0, 128)
        g = random.randint(0, 128)
        b = random.randint(0, 128)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def visualize_colors(self, data_name, save_path):
        fig, ax = plt.subplots()
        x = range(self.num_class)
        ax.barh(x, [1]*self.num_class, color=self.light_colors)
        ax.barh(x, [-1]*self.num_class, color=self.deep_colors)
        ax.axis('off')
        file_name = f'color_chart_{self.num_class}.png'
        file_path = os.path.join(save_path, "data", data_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        plt.close(fig)  

    def convert_graph_to_image(self, edge_index, file_path, node_labels,
                               store_flag='vary', color_style='deep', layout_aug=False):
        if store_flag == 'vary' or not os.path.exists(file_path):
            file_path = file_path.split('.png')[0]

            # Check if the path of dirname exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if layout_aug:
                dot = graphviz.Digraph(format='png', engine=random.choice(self.layout_list))
            else:
                dot = graphviz.Digraph(format='png', engine='sfdp')

            unique_nodes = torch.unique(edge_index).cpu().numpy()

            dot.attr('edge', arrowsize='0.25')

            for node in unique_nodes:
                color_index = node_labels[node]
                if color_index != -1:
                    if color_style == 'light':
                        light_color = self.light_colors[color_index] 
                        dot.node(str(node), shape='box', color=light_color, style='filled')
                    else:
                        deep_color = self.deep_colors[color_index]
                        dot.node(str(node), shape='box', color=deep_color)
                else:
                    dot.node(str(node), shape='box')
            # Add edges, using subgraph labels
            edge_index = edge_index.t().tolist()
            for start, end in edge_index:
                dot.edge(str(start), str(end))

            dot.render(filename=file_path, cleanup=True)


class GraphDescriber(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.graph_template = (
            "In a directed graph, (i,j) means that node i and node j are connected with a directed edge."
            " The nodes are numbered from [P] to [P], and the edges are:\n"
        )
        self.edge_template = "([P],[P])\n"
        self.node_class_template = "Each node in the graph is assigned to one of [P] classes, we have known:\n"
        self.node_template = "Node [P] belongs to Class [P]\n"
        
    def convert_graph_to_description(self, edge_index, node_labels):
        unique_nodes = torch.unique(edge_index).cpu().numpy()
        num_nodes = len(node_labels)
        graph_description = (self.graph_template.replace('[P]', '0', 1)
                             .replace('[P]', str(num_nodes-1), 1))
        edge_index = edge_index.t().tolist()
        for start, end in edge_index:
            graph_description += (self.edge_template.replace('[P]', str(start), 1)
                                  .replace('[P]', str(end), 1))
        
        graph_description += self.node_class_template.replace('[P]', str(self.num_class), 1)
        for node in unique_nodes:
            if node_labels[node] != -1:
                graph_description += (self.node_template.replace('[P]', str(node), 1)
                                      .replace('[P]', str(node_labels[node]), 1))
        return graph_description
    
    
class Questioner(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.visual_graph_template = (
            "<image>\nThis image depicts a directed graph composed of box nodes labeled with numbers, and arrows"
            " representing edges between nodes. There are total [P] classes, and each node belongs to one of them."
            " Nodes of known class depicted in specific colors,"
            " while nodes with unknown class are represented in white.\n"
        )
        self.visual_text_transition = "The following paragraph describes the same graph as the image: "
        self.task_responsibility = (
            "The task is semi-supervised node classification, and needs to predict which class Node 0 belongs to,"
            " based on graph structure and known node classes."
        )
        self.output_specification = "Q: Node 0 belongs to Class:"
        
    def generate_vt_query(self, graph_description):
        query = ""
        query += self.visual_graph_template.replace('[P]', str(self.num_class), 1)
        query += self.visual_text_transition
        query += graph_description
        query += self.task_responsibility
        query += self.output_specification
        return query
    
    
class DataConstructor(object):
    def __init__(self, data_name, save_path="./"):
        self.save_path = save_path
        self.data_name = data_name
        if data_name in ['Cora', 'CiteSeer']:
            my_data = MyPlanetoidDataset(data_name=data_name, save_path=save_path)
            sample_sizes = [5, 5]
        elif data_name in ['PolBlogs']:
            my_data = MyPolblogsDataset(save_path=save_path)
            sample_sizes = [5, 5]
        elif data_name in ['email-Eu-core']:
            my_data = MyEmailEuCoreDataset(save_path=save_path)
            sample_sizes = [5, 5]
        else:
            raise NotImplementedError("Do not support this type of data.")

        self.data = my_data.data
        self.num_classes = my_data.dataset.num_classes
        self.train_loader = NeighborSampler(
            self.data.edge_index, node_idx=self.data.train_mask,
            sizes=sample_sizes, batch_size=1, shuffle=True,
            num_workers=12, worker_init_fn=worker_init_fn
        )
        self.test_loader = NeighborSampler(
            self.data.edge_index, node_idx=self.data.test_mask,
            sizes=sample_sizes, batch_size=1, shuffle=True,
            num_workers=4, worker_init_fn=worker_init_fn
        )
        
        self.graph_visualizer = GraphVisualizer(self.num_classes, data_name=data_name, save_path=save_path)
        self.graph_describer = GraphDescriber(self.num_classes)
        self.questioner = Questioner(self.num_classes)
        
    def construct_json(self, data_split='train'):
        train_samples = []
        assert data_split == 'train' or data_split == 'test'
        loader = self.train_loader if data_split == 'train' else self.test_loader
        for batch_size, n_id, adjs in tqdm(loader):
            n_id = n_id.tolist()
            
            sample = {}
            image_path = f"data/{self.data_name}/image/{data_split}/subgraph_image_{n_id[0]}.png"
            node_labels = -1 * np.ones((len(n_id),), dtype=np.int64)

            for local_idx, global_idx in enumerate(n_id):
                if self.data.train_mask[global_idx]:
                    node_labels[local_idx] = self.data.y[global_idx]
            node_labels[0] = -1

            # Save visual graphs to path
            self.graph_visualizer.convert_graph_to_image(
                adjs[0].edge_index, os.path.join(self.save_path, image_path), node_labels,
                'vary', 'light', False
            )
            graph_description = self.graph_describer.convert_graph_to_description(adjs[0].edge_index, node_labels)
            
            query = self.questioner.generate_vt_query(graph_description)

            # Write to the json file
            sample['id'] = self.data_name + '-' + str(n_id[0])
            sample['image'] = image_path
            sample["conversations"] = []
            sample["conversations"].append({
                "from": "human",
                "value": query
            })
            sample["conversations"].append({
                "from": "gpt",
                "value": str(self.data.y[n_id[0]].item())
            })
            
            train_samples.append(sample)
        with open(os.path.join(self.save_path, f'data/{self.data_name}/Vision_Text_{data_split}.json'), 'w') as f:
            json.dump(train_samples, f, indent=4)
            

if __name__ == "__main__":   
    # Initialization: Generate all the data once
    # Use: Call data_constructor_cora.construct_json() for data construction before each epoch
    seed_torch()
    data_constructor_cora = DataConstructor(data_name='Cora', save_path="../dataset/Node_Cls")
    data_constructor_cora.construct_json(data_split="train")
    data_constructor_cora.construct_json(data_split="test")

    data_constructor_citeseer = DataConstructor('CiteSeer', save_path="../dataset/Node_Cls")
    data_constructor_citeseer.construct_json(data_split="train")
    data_constructor_citeseer.construct_json(data_split="test")

    data_constructor_polblogs = DataConstructor('PolBlogs', save_path="../dataset/Node_Cls")
    data_constructor_polblogs.construct_json(data_split="train")
    data_constructor_polblogs.construct_json(data_split="test")

    data_constructor_emaileucore = DataConstructor('email-Eu-core', save_path="../dataset/Node_Cls")
    data_constructor_emaileucore.construct_json(data_split="train")
    data_constructor_emaileucore.construct_json(data_split="test")
