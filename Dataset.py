import pandas as pd
from dgl.data import DGLDataset
import pickle
import torch
import dgl
from tqdm import tqdm
import os
import numpy as np
from utils import encode_map


class MyDataset(DGLDataset):
    def __init__(self,
                 name=None,
                 url=None,
                 raw_dir="./raw_dir",
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 type=None):
        self.event_name = name
        self.type = type
        super(MyDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        pass

    def process(self):
        raw_dir_new = self.raw_dir + '/' + self.event_name
        self.hetro_graph, self.homo_graph, self.label = self._load_graph(raw_dir_new)

    def _load_graph(self, path):
        type = self.type
        load_hetro_path = os.path.join(path, f"Graph/{type}_hetro_subgraph")
        list_hetro_subgraph, _ = dgl.load_graphs(load_hetro_path)
        load_homo_path = os.path.join(path, f"Graph/{type}_homo_subgraph")
        list_homo_subgraph, _ = dgl.load_graphs(load_homo_path)
        seq_labels_path = os.path.join(path, f"{type}_seq_labels.npy")
        seq_labels = np.load(seq_labels_path, allow_pickle=True)
        seq_labels = torch.tensor(seq_labels, dtype=torch.int64)
        return list_hetro_subgraph, list_homo_subgraph, seq_labels

    def __getitem__(self, idx):
        return self.hetro_graph[idx], self.homo_graph[idx], self.label[idx]

    def __len__(self):
        return len(self.hetro_graph)
