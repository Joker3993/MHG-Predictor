import dgl
import numpy as np
import torch
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings(action='ignore')
from utils import *
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
import pandas as pd
import pickle
from tqdm import tqdm


class Whole_Graph:
    def __init__(self, eventlog, n_bins):
        self._eventlog = eventlog
        self._bin = n_bins
        self._fold = 0

    def edges_process(self, event_log):

        groupProcessList = []

        for groupId, group in tqdm(event_log.groupby('case')):
            group = pd.concat([group.iloc[0:1], group])

            group['timestamp'] = pd.to_datetime(group['timestamp'])

            group['duration'] = group['timestamp'].diff().dt.total_seconds()[1:]

            group['duration'] = group['duration'].values / 86400
            group['duration'] = group['duration'].shift(-1)

            group = group.shift(periods=-1)

            group = group.iloc[:-1, :]
            group['duration'] = group['duration'].fillna(1)

            groupProcessList.append(group)

        edges_raw = pd.concat(groupProcessList)

        edges_raw.index = range(len(edges_raw))

        return edges_raw

    def save_node_feature(self, nodes_raw, type):
        node_name = []
        node_name.append('eventlog')

        for col in nodes_raw.columns.tolist():
            if col == "id":
                continue
            elif col == 'case':
                node_feature = nodes_raw[col].tolist()
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + col + ".npy",
                          'wb') as file:
                    pickle.dump(node_feature, file)
            else:
                node_feature = nodes_raw[col].tolist()
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + col + ".npy",
                          'wb') as file:
                    pickle.dump(node_feature, file)
            if col == 'duration_features' or col == 'node_col' or col == 'timestamp':
                continue
            else:
                node_name.append(col)

        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + "node_name" + ".npy", 'wb') as file:
            pickle.dump(node_name, file)
        print(node_name)

    def save_edges(self, edges_raw, type):

        for col in edges_raw.columns.tolist():

            """根据名为"case"的列在名为edges_raw的DataFrame上进行分组。
            然后，它应用了agg函数来对每个组中的值进行聚合。聚合函数使用了一个lambda函数，将指定列（col）中的每个组的值转换为一个列表。"""
            sequence = edges_raw.groupby("case", sort=False).agg({col: lambda x: list(x)})

            if col == "node_col":
                list_src, list_dst, list_case = create_activity_activity(sequence)

                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "list_src" + ".npy",
                          'wb') as file:
                    pickle.dump(list_src, file)
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "list_dst" + ".npy",
                          'wb') as file:
                    pickle.dump(list_dst, file)
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "list_case" + ".npy",
                          'wb') as file:
                    pickle.dump(list_case, file)
            elif col == "duration":
                duration_labels = edges_raw['duration'].tolist()
                with open("raw_dir/" + self._eventlog + "_" + str(
                        self._fold) + "/" + type + '_' + col + '_labels' + ".npy", 'wb') as file:
                    pickle.dump(duration_labels, file)

    def create_hetero_graph(self, train_df, type):

        train_df['node_col'] = range(len(train_df))

        """案例列需要再次整数编码"""
        col = 'case'

        att_encode_map = encode_map(set(train_df[col].values))

        train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))

        event_list = [0] * len(train_df['case'].unique().tolist())

        self.save_edges(train_df, type)
        """多层异构图的创建,活动层的同构关系还没有加进去"""
        dict_heterograph = {
            ('case', 'CaseToLog', 'eventlog'): (train_df['case'].unique().tolist(), event_list),
            ('activity', 'ActivityToCase', 'case'): (train_df['node_col'].tolist(), train_df['case'].tolist()),
            ('activity', 'next', 'activity'): ([0], [0]),
        }
        for col in train_df.columns.tolist():
            if col == "case" or col == "node_col" or col == "activity" or col == "timestamp" or col == "duration_features":
                pass
            else:
                dict_heterograph[(col, 'include', 'activity')] = (
                    train_df['node_col'].tolist(), train_df['node_col'].tolist())
        hetero_graph = dgl.heterograph(dict_heterograph)

        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + "etypes" + ".npy", 'wb') as file:
            pickle.dump(hetero_graph.etypes, file)
        print("异构图边类型：", hetero_graph.etypes)
        path = "./raw_dir" + "/" + self._eventlog + "_" + str(self._fold)

        source_path = os.path.join(path, f"{type}_list_src.npy")
        destination_path = os.path.join(path, f"{type}_list_dst.npy")

        features_path = os.path.join(path, "feature_list.npy")
        case_path = os.path.join(path, f"{type}_list_case.npy")

        source_lists = np.load(source_path, allow_pickle=True)
        destination_lists = np.load(destination_path, allow_pickle=True)
        case_lists = np.load(case_path, allow_pickle=True)
        feature_lists = np.load(features_path, allow_pickle=True)

        node_features = {}
        for feature_name in feature_lists:
            feature_path = os.path.join(path, f"{type}_{feature_name}.npy")
            att_lists = np.load(feature_path, allow_pickle=True)
            node_features[feature_name] = att_lists

        for feature_name, feature_data in node_features.items():
            if feature_name == 'duration_features':
                hetero_graph.nodes['duration'].data['duration'] = torch.tensor(feature_data, dtype=torch.int64)
            elif feature_name == 'case':
                hetero_graph.nodes[f'{feature_name}'].data[feature_name] = torch.tensor(list(set(feature_data)),
                                                                                        dtype=torch.int64)
            else:

                if f'{feature_name}' in hetero_graph.ntypes:
                    hetero_graph.nodes[f'{feature_name}'].data[feature_name] = torch.tensor(feature_data,
                                                                                            dtype=torch.int64)
                else:

                    print(f"Node type '{feature_name}' does not exist. You may need to create it.")

        hetero_graph.nodes['eventlog'].data['eventlog'] = torch.tensor([0], dtype=torch.int64)

        for i in tqdm(range(len(case_lists))):
            hetero_graph.add_edges(torch.tensor(source_lists[i]), torch.tensor(destination_lists[i]),
                                   etype=('activity', 'next', 'activity'))
        return hetero_graph

    def whole_main(self):
        for fold in range(3):
            self._fold = fold
            print(
                f"--------------------------------------{self._eventlog}---第{self._fold}折开始-------------------------------------------")

            df_train = pd.read_csv(
                "fold/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(self._fold) + "_train.csv",
                sep=',',
                header=0, index_col=False)
            df_test = pd.read_csv(
                "fold/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(self._fold) + "_test.csv",
                sep=',',
                header=0, index_col=False)

            np.random.seed(133)
            grouped = df_train.groupby('case')
            new_order = np.random.permutation(list(grouped.groups.keys()))
            new_groups = [grouped.get_group(key) for key in new_order]
            log_shuffled = pd.concat(new_groups)
            log_shuffled.index = range(len(log_shuffled))

            train, valid = train_test_split(log_shuffled, test_size=0.2, shuffle=False)
            train.index = range(len(train))
            valid.index = range(len(valid))
            path = "raw_dir/" + self._eventlog + "_" + str(self._fold)

            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            train.to_csv("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_train.csv",
                         index=False)
            valid.to_csv("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_valid.csv",
                         index=False)
            df_test.to_csv("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_test.csv",
                           index=False)

            train_df = pd.read_csv(
                "raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_train.csv",
                sep=',',
                header=0, index_col=False)
            val_df = pd.read_csv(
                "raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_valid.csv",
                sep=',',
                header=0, index_col=False)
            test_df = pd.read_csv(
                "raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_test.csv",
                sep=',',
                header=0, index_col=False)
            feature_list = []

            for col in train_df.columns.tolist():

                if col == "timestamp":
                    continue

                train_df[col].fillna(method='ffill', inplace=True)
                val_df[col].fillna(method='ffill', inplace=True)
                test_df[col].fillna(method='ffill', inplace=True)

                total_data = pd.concat([train_df, val_df, test_df])

                att_encode_map = encode_map(set(total_data[col].values))

                train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
                val_df[col] = val_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
                test_df[col] = test_df[col].apply(lambda e: att_encode_map.get(str(e), -1))

                total_data = pd.concat([train_df, val_df, test_df])

                feature_list.append(col)
            print('111\n', total_data)
            feature_list.append('duration_features')

            with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + 'feature' + '_' + "list" + ".npy",
                      'wb') as file:
                pickle.dump(feature_list, file)
            """单独操作一下，持续时间节点的特征"""
            train_df['node_col'] = range(len(train_df))
            val_df['node_col'] = range(len(val_df))
            test_df['node_col'] = range(len(test_df))
            train_df = self.edges_process(train_df)
            val_df = self.edges_process(val_df)
            test_df = self.edges_process(test_df)
            X1 = train_df['duration'].to_numpy().reshape(-1, 1)
            X2 = val_df['duration'].to_numpy().reshape(-1, 1)
            X3 = test_df['duration'].to_numpy().reshape(-1, 1)
            change = KBinsDiscretizer(n_bins=self._bin, encode='ordinal', strategy='quantile')
            change.fit(X1)
            train_df['duration_features'] = change.transform(X1)
            val_df['duration_features'] = change.transform(X2)
            test_df['duration_features'] = change.transform(X3)
            total_data = pd.concat([train_df, val_df, test_df])
            for col in train_df.columns.tolist():
                total_data = pd.concat([train_df, val_df, test_df])

                att_count = len(total_data[col].unique())
                print(f"{col}:{att_count}")

                if col == 'duration_features':
                    with open("raw_dir/" + self._eventlog + "_" + str(
                            self._fold) + "/" + 'duration' + '_' + "info" + ".npy", 'wb') as file:
                        pickle.dump(att_count, file)

                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + col + '_' + "info" + ".npy",
                          'wb') as file:
                    pickle.dump(att_count, file)
                with open(
                        'raw_dir/' + self._eventlog + "_" + str(self._fold) + '/' + 'eventlog' + '_' + 'info' + '.npy',
                        'wb') as file:
                    pickle.dump(1, file)
            print('外面的\n', train_df)
            self.save_node_feature(train_df, type='train')
            self.save_node_feature(val_df, type='val')
            self.save_node_feature(test_df, type='test')

            train_hetro_graph = self.create_hetero_graph(train_df, type='train')
            val_hetro_graph = self.create_hetero_graph(val_df, type='val')
            test_hetro_graph = self.create_hetero_graph(test_df, type='test')
            dgl.save_graphs(f"raw_dir/{self._eventlog}" + "_" + str(self._fold) + "/Graph/train_hetro_graph",
                            [train_hetro_graph])
            dgl.save_graphs(f"raw_dir/{self._eventlog}" + "_" + str(self._fold) + "/Graph/val_hetro_graph",
                            [val_hetro_graph])
            dgl.save_graphs(f"raw_dir/{self._eventlog}" + "_" + str(self._fold) + "/Graph/test_hetro_graph",
                            [test_hetro_graph])
