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
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """
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
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        raw_dir_new = self.raw_dir + '/' + self.event_name
        self.hetro_graph,self.homo_graph, self.label = self._load_graph(raw_dir_new)

    def _load_graph(self, path):
        # 从类中获取 type 属性（假设它在其他地方定义）
        type = self.type

        """加载整图"""
        load_hetro_path = os.path.join(path, f"Graph/{type}_hetro_subgraph")
        list_hetro_subgraph, _ = dgl.load_graphs(load_hetro_path)  # 加载了一个存储着图的图列表

        load_homo_path = os.path.join(path, f"Graph/{type}_homo_subgraph")
        list_homo_subgraph, _ = dgl.load_graphs(load_homo_path)  # 加载了一个存储着图的图列表

        seq_labels_path = os.path.join(path, f"{type}_seq_labels.npy")
        seq_labels = np.load(seq_labels_path, allow_pickle=True)
        seq_labels = torch.tensor(seq_labels, dtype=torch.int64)

        return list_hetro_subgraph, list_homo_subgraph, seq_labels


    def __getitem__(self, idx):
        """ 通过idx获取对应的图和标签

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.hetro_graph[idx], self.homo_graph[idx],self.label[idx]

    def __len__(self):
        # 数据样本的数量
        return len(self.hetro_graph)

