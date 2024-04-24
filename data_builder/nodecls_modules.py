import os
import torch
from torch_geometric.datasets import PolBlogs
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
from torch.backends import cudnn
import random
import numpy as np


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    seed_torch(worker_seed % (2**32))


class MyPolblogsDataset(object):
    def __init__(self, save_path="./", transform=None):
        self.dataset = PolBlogs(root=os.path.join(save_path, 'dataset/Polblogs'), transform=transform)
        self.data = self.dataset[0]
        self.num_classes = 2  # PolBlogs has two classes
        
        # Planetoid: directed (or undirected graph, if possible)
        # self.data.edge_index = to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)

        self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        
        # Split: 8:1:1
        nodes = range(self.data.num_nodes)
        train_nodes, test_nodes = train_test_split(nodes, test_size=0.2, random_state=42)
        val_nodes, test_nodes = train_test_split(test_nodes, test_size=0.5, random_state=42)
        
        self.data.train_mask[train_nodes] = True
        self.data.val_mask[val_nodes] = True
        self.data.test_mask[test_nodes] = True

    def get(self):
        return self.data


class MyPlanetoidDataset(object):
    def __init__(self, data_name, save_path="./"):
        self.dataset = Planetoid(root=os.path.join(save_path, 'dataset/' + data_name),
                                 name=data_name, transform=NormalizeFeatures())
        self.data = self.dataset[0]

        # Create train/val/test mask
        self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        
        # Split: 8:1:1
        nodes = range(self.data.num_nodes)
        train_nodes, test_nodes = train_test_split(nodes, test_size=0.2, random_state=42)
        val_nodes, test_nodes = train_test_split(test_nodes, test_size=0.5, random_state=42)
        
        self.data.train_mask[train_nodes] = True
        self.data.val_mask[val_nodes] = True
        self.data.test_mask[test_nodes] = True

    def get(self):
        return self.data


class MyEmailEuCoreDataset(object):
    def __init__(self, save_path="./"):
        self.save_path = save_path
        # Initialize the dataset and data objects
        self.dataset = type('Dataset', (object,), {})()
        self.data = type('Data', (object,), {})()
        
        self.dataset.num_classes = 7
        self.data.num_nodes = self.get_num_nodes()  # Suppose there is a function to get the number of nodes

        self.data.edge_index = torch.tensor(self.read_edges(), dtype=torch.long).t().contiguous()
        self.data.y = torch.tensor(self.read_labels(), dtype=torch.long)

        # Create train/val/test mask
        self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        
        # Split: 8:1:1
        nodes = list(range(self.data.num_nodes))
        train_nodes, test_nodes = train_test_split(nodes, test_size=0.2, random_state=42)
        val_nodes, test_nodes = train_test_split(test_nodes, test_size=0.5, random_state=42)
        
        self.data.train_mask[train_nodes] = True
        self.data.val_mask[val_nodes] = True
        self.data.test_mask[test_nodes] = True

    def read_edges(self):
        edge_list = []
        with open(os.path.join(self.save_path, 'dataset/email-Eu-core/email-Eu-core.txt'), 'r') as f:
            for line in f:
                node1, node2 = map(int, line.split())
                edge_list.append([node1, node2])
        return edge_list

    def read_labels(self):
        label_list = []
        with open(os.path.join(self.save_path, 'dataset/email-Eu-core/email-Eu-core-department-labels.txt'), 'r') as f:
            for line in f:
                node, label = map(int, line.split())
                label_list.append(label // 6)
        return label_list

    @staticmethod
    def get_num_nodes():
        return 1005

    def get(self):
        return self.data


if __name__ == "__main__":
    seed_torch()
