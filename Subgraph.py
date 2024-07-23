import os
import pickle
import dgl
import dgl.heterograph
import numpy as np
import torch
import dgl
import numpy as np
import pandas as pd
import pandas as pd
from tqdm import tqdm
from utils import encode_map


class SubGraph:
    def __init__(self, eventlog):
        self._eventlog = eventlog
        self._fold = 0

    def get_sequences_activity(self, df):
        sequences = []
        cases = []
        groups = df.groupby('case')
        for case, group in groups:
            activities = group['node_col'].tolist()
            for i in range(len(activities) - 1):
                prefix = activities[:i + 1]
                sequences.append(prefix)
                cases.append(case)
        return sequences, cases

    def get_sequences_time(self, df):
        sequences = []
        cases = []
        labels = []
        groups = df.groupby('case')
        for case, group in groups:
            node_cols = group['node_col'].tolist()
            durations = group['activity'].tolist()
            for i in range(len(node_cols) - 1):
                prefix = node_cols[:i + 1]
                sequences.append(prefix)
                cases.append(case)
                label = durations[i + 1]
                labels.append(label)
        return sequences, cases, labels

    def subgraph(self, data, loaded_graphs, type):
        data['node_col'] = range(len(data))

        col = 'case'
        att_encode_map = encode_map(set(data[col].values))
        data[col] = data[col].apply(lambda e: att_encode_map.get(str(e), -1))
        seq_activity, seq_case1 = self.get_sequences_activity(data)
        seq_duration, seq_case2, seq_labels = self.get_sequences_time(data)
        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_activity" + ".npy",
                  'wb') as file:
            pickle.dump(seq_activity, file)
        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_case1" + ".npy",
                  'wb') as file:
            pickle.dump(seq_case1, file)
        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_labels" + ".npy",
                  'wb') as file:
            pickle.dump(seq_labels, file)
        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_duration" + ".npy",
                  'wb') as file:
            pickle.dump(seq_duration, file)
        list_hetro_subgraph = []
        list_homo_subgraph = []

        for i in tqdm(range(len(seq_case1))):
            node_dict = {
                'eventlog': [0],
                'case': seq_case1[i]
            }
            for ntype in loaded_graphs[0].ntypes:
                if ntype == 'case' or ntype == 'eventlog':
                    continue
                elif ntype == 'duration':
                    node_dict[ntype] = seq_duration[i]
                else:
                    node_dict[ntype] = seq_activity[i]
            hetro_subgraph = dgl.node_subgraph(loaded_graphs[0], nodes=node_dict)
            list_hetro_subgraph.append(hetro_subgraph)

        for i in tqdm(range(len(seq_case1))):
            homo_subgraph = dgl.node_subgraph(loaded_graphs[0], {"activity": seq_activity[i]})
            if homo_subgraph.num_edges('next') == 0:
                homo_subgraph = dgl.add_self_loop(homo_subgraph, etype='next')
            list_homo_subgraph.append(dgl.to_homogeneous(homo_subgraph, ndata=['activity']))
        return list_hetro_subgraph, list_homo_subgraph

    def Subgraph_main(self):
        for fold in range(3):
            self._fold = fold
            print(
                f"--------------------------------------{self._eventlog}第{self._fold}折开始-------------------------------------------")
            for type in ['train', 'test', 'val']:

                load_path = "raw_dir/" + self._eventlog + "_" + str(self._fold) + f"/Graph/{type}_hetro_graph"
                loaded_graphs, _ = dgl.load_graphs(load_path)
                path = "./raw_dir" + "/" + self._eventlog + "_" + str(self._fold)
                features_path = os.path.join(path, "feature_list.npy")
                feature_lists = np.load(features_path, allow_pickle=True)
                feature_lists.append('case')
                feature_lists.append('duration')
                node_features = {}
                for feature_name in feature_lists:
                    if feature_name == 'duration':
                        feature_path = os.path.join(path, f"{type}_{feature_name}_labels.npy")
                        att_lists = np.load(feature_path, allow_pickle=True)
                        node_features[feature_name] = att_lists
                    else:
                        feature_path = os.path.join(path, f"{type}_{feature_name}.npy")
                        att_lists = np.load(feature_path, allow_pickle=True)
                        node_features[feature_name] = att_lists
                data = pd.DataFrame(node_features)
                train_hetro_subgraph, train_homo_subgraph = self.subgraph(data, loaded_graphs, type)
                print(data)
                dgl.save_graphs("raw_dir/" + self._eventlog + "_" + str(self._fold) + f"/Graph/{type}_hetro_subgraph",
                                train_hetro_subgraph)
                dgl.save_graphs("raw_dir/" + self._eventlog + "_" + str(self._fold) + f"/Graph/{type}_homo_subgraph",
                                train_homo_subgraph)
                print(self._eventlog + "_" + str(self._fold) + "_" + f'{type}数据集子图生成完毕\n')
