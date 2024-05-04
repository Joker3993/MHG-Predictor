import dgl
import numpy as np
import torch
from sklearn.cluster import KMeans

import warnings

from tqdm import tqdm
import pandas as pd
warnings.filterwarnings(action='ignore')  # 忽略警告
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

    def edges_process(self,event_log):
        # 存储处理过的分组数据的列表
        groupProcessList = []

        # 遍历事件日志中基于 'case' 列分组的数据
        for groupId, group in tqdm(event_log.groupby('case')):
            # 复制第一行在每个分组的开头，用于计算持续时间
            group = pd.concat([group.iloc[0:1], group])

            """增加持续时间的特征"""
            # 将 'timestamp' 列转换为datetime对象
            group['timestamp'] = pd.to_datetime(group['timestamp'])
            # print('原始时间戳\n',group['timestamp'])

            # 计算活动持续时间，并将结果保存在 'duration' 列中
            group['duration'] = group['timestamp'].diff().dt.total_seconds()[1:]
            # print('还没缩放过的持续时间\n',group['duration'])
            group['duration'] = group['duration'].values / 86400

            group['duration'] = group['duration'].shift(-1)#时间差这个要上移2行，所以这里提前先上移一行
            # 将DataFrame中的每一列元素向上移动一行
            group = group.shift(periods=-1)

            # 去除最后一行，避免多余的空值
            group = group.iloc[:-1, :]
            group['duration'] = group['duration'].fillna(1)  # 将 NaN 值（最后一行的持续时间）设置为 1

            # 将处理过的分组数据添加到列表中
            groupProcessList.append(group)

        # 将所有处理过的分组数据合并为一个DataFrame

        edges_raw = pd.concat(groupProcessList)
        # print('检查时间\n',edges_raw)

        # 重新索引DataFrame
        edges_raw.index = range(len(edges_raw))

        # 返回处理过的边数据
        return edges_raw

    def save_node_feature(self,nodes_raw, type):
        node_name = []
        node_name.append('eventlog')

        # 循环保存节点特征
        for col in nodes_raw.columns.tolist():
            if  col == "id":
                continue
            elif col == 'case':

                node_feature = nodes_raw[col].tolist()

                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + col + ".npy", 'wb') as file:
                    pickle.dump(node_feature, file)
            else:

                node_feature = nodes_raw[col].tolist()
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + col + ".npy", 'wb') as file:
                    pickle.dump(node_feature, file)

            if col == 'duration_features' or col == 'node_col' or col == 'timestamp':
                continue
            else:
                node_name.append(col)

        #单独存一下节点名称
        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + "node_name" + ".npy", 'wb') as file:
            pickle.dump(node_name, file)

        print(node_name)

    def save_edges(self,edges_raw, type):
        # 循环遍历边数据框的列
        for col in edges_raw.columns.tolist():
            # 根据 'case' 列分组，获取序列数据
            """根据名为"case"的列在名为edges_raw的DataFrame上进行分组。
            然后，它应用了agg函数来对每个组中的值进行聚合。聚合函数使用了一个lambda函数，将指定列（col）中的每个组的值转换为一个列表。"""
            sequence = edges_raw.groupby("case", sort=False).agg({col: lambda x: list(x)})

            # 根据列名选择不同的处理方式
            if col == "node_col":

                list_src,list_dst,list_case = create_activity_activity(sequence)

                # 将序列、标签和案例列表保存为Numpy数组
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "list_src" + ".npy", 'wb') as file:
                    pickle.dump(list_src, file)
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "list_dst" + ".npy", 'wb') as file:
                    pickle.dump(list_dst, file)
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "list_case" + ".npy", 'wb') as file:
                    pickle.dump(list_case, file)

            elif col == "duration":

                duration_labels = edges_raw['duration'].tolist()
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + col +'_labels'+".npy", 'wb') as file:
                    pickle.dump(duration_labels, file)

    def create_hetero_graph(self,train_df,type):
        #三个数据集添加“node_col”列，这个列专门用来构建图时当做节点的编号来用
        train_df['node_col'] = range(len(train_df))
        # print(train_df)

        """案例列需要再次整数编码"""
        col = 'case'
        # 使用encode_map对该列进行编码
        att_encode_map = encode_map(set(train_df[col].values))
        # 利用上面生成的字典，将各列对应字符转换为整数
        train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))

        #日志节点的编码列表创建
        event_list = [0] * len(train_df['case'].unique().tolist())

        #edge_process,在main函数里处理过了，所以这里直接就跳过了
        # ac_ac = edges_process(train_df)
        self.save_edges(train_df ,type)


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


        #单独存一下所有边类型
        with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + "etypes" + ".npy", 'wb') as file:
            pickle.dump(hetero_graph.etypes, file)

        print("异构图边类型：",hetero_graph.etypes)

        path = "./raw_dir" + "/" + self._eventlog + "_" + str(self._fold)

        # 设置源，目标，标签，特征和案例信息的文件路径
        source_path = os.path.join(path, f"{type}_list_src.npy")
        destination_path = os.path.join(path, f"{type}_list_dst.npy")
        # label_path = os.path.join(path, f"{type}_label.npy")
        features_path = os.path.join(path, "feature_list.npy")
        case_path = os.path.join(path, f"{type}_list_case.npy")

        # 加载源，目标，标签，特征和案例信息
        source_lists = np.load(source_path, allow_pickle=True)
        destination_lists = np.load(destination_path, allow_pickle=True)
        case_lists = np.load(case_path, allow_pickle=True)
        feature_lists = np.load(features_path, allow_pickle=True)



        # 为所有图加载节点特征
        node_features = {}
        for feature_name in feature_lists:
            feature_path = os.path.join(path, f"{type}_{feature_name}.npy")
            att_lists = np.load(feature_path, allow_pickle=True)
            node_features[feature_name] = att_lists

        # 将节点特征添加到图中
        for feature_name, feature_data in node_features.items():
            if feature_name == 'duration_features':
                hetero_graph.nodes['duration'].data['duration'] = torch.tensor(feature_data, dtype=torch.int64)
            elif feature_name == 'case':
                hetero_graph.nodes[f'{feature_name}'].data[feature_name] = torch.tensor(list(set(feature_data)), dtype=torch.int64)
            else:
                # 使用节点类型为特征名存储其他特征
                if f'{feature_name}' in hetero_graph.ntypes:
                    hetero_graph.nodes[f'{feature_name}'].data[feature_name] = torch.tensor(feature_data, dtype=torch.int64)
                else:
                    # 处理节点类型不存在的情况，可能需要创建新的节点类型
                    print(f"Node type '{feature_name}' does not exist. You may need to create it.")


        #给日志节点加一个特征。
        hetero_graph.nodes['eventlog'].data['eventlog'] = torch.tensor([0],dtype=torch.int64)

        # 批量往异构图中添加活动层的同构关系的边
        for i in tqdm(range(len(case_lists))):
            hetero_graph.add_edges(torch.tensor(source_lists[i]),torch.tensor(destination_lists[i]),
                                   etype=('activity', 'next', 'activity'))
        return hetero_graph


    def whole_main(self):
        for fold in range(3):
            self._fold = fold
            print(f"--------------------------------------{self._eventlog}---第{self._fold}折开始-------------------------------------------")

            # 读取训练和测试数据集
            df_train = pd.read_csv("fold/" + self._eventlog + "/" + self._eventlog +"_kfoldcv_"+ str(self._fold) + "_train.csv",
                                   sep=',',
                                   header=0, index_col=False)
            df_test = pd.read_csv("fold/" + self._eventlog + "/" + self._eventlog +"_kfoldcv_"+ str(self._fold) + "_test.csv",
                                  sep=',',
                                  header=0, index_col=False)

            # 随机排列训练数据
            np.random.seed(133)
            grouped = df_train.groupby('case')
            new_order = np.random.permutation(list(grouped.groups.keys()))
            new_groups = [grouped.get_group(key) for key in new_order]
            log_shuffled = pd.concat(new_groups)
            log_shuffled.index = range(len(log_shuffled))

            # 拆分训练数据集为训练集和验证集
            train, valid = train_test_split(log_shuffled, test_size=0.2, shuffle=False)
            train.index = range(len(train))
            valid.index = range(len(valid))

            path="raw_dir/" + self._eventlog + "_" + str(self._fold)
            # 创建存储数据的目录（如果不存在）
            if not os.path.exists(path):
                # os.mkdir(path)
                os.makedirs(path, exist_ok=True)

            # 保存训练集、验证集和测试集为 CSV 文件
            train.to_csv("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + self._eventlog + "_train.csv", index=False)
            valid.to_csv("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + self._eventlog + "_valid.csv", index=False)
            df_test.to_csv("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + self._eventlog + "_test.csv", index=False)

            # 读取保存的训练集、验证集和测试集数据
            train_df = pd.read_csv("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + self._eventlog + "_train.csv",
                                   sep=',',
                                   header=0, index_col=False)
            val_df = pd.read_csv("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + self._eventlog + "_valid.csv",
                                 sep=',',
                                 header=0, index_col=False)
            test_df = pd.read_csv("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + self._eventlog + "_test.csv",
                                  sep=',',
                                  header=0, index_col=False)

            feature_list = []

            # 遍历训练集的所有列,对各个列进行整数编码
            for col in train_df.columns.tolist(): # 每个数据集里的列都是一样的
                # 忽略 "case" 和 "timestamp" 列
                if col == "timestamp":
                    continue

                # 使用前向填充方法填充缺失值
                train_df[col].fillna(method='ffill', inplace=True)
                val_df[col].fillna(method='ffill', inplace=True)
                test_df[col].fillna(method='ffill', inplace=True)

                # 将训练集、验证集和测试集合并成一个总的数据集
                total_data = pd.concat([train_df, val_df, test_df])

                # 使用encode_map对该列进行编码
                att_encode_map = encode_map(set(total_data[col].values))
                #利用上面生成的字典，将各列对应字符转换为整数
                train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
                val_df[col] = val_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
                test_df[col] = test_df[col].apply(lambda e: att_encode_map.get(str(e), -1))

                # 将训练集、验证集和测试集合并成一个总的数据集
                total_data = pd.concat([train_df, val_df, test_df])

                # 将列名添加到特征列表中
                feature_list.append(col)

            print('111\n', total_data)
            feature_list.append('duration_features')

            # feature_list.remove('timestamp')
            # 将特征列表保存为 Numpy 文件
            with open("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + 'feature' + '_' + "list" + ".npy", 'wb') as file:
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

                # 计算每列数据的唯一值个数保存下来，后续模型中嵌入维度的设置需要用到
                # att_count = len(set(total_data[col].values))
                att_count = len(total_data[col].unique())
                print(f"{col}:{att_count}")

                # 由于一些地方的命名问题，所以duration_features才是真正用到的持续时间特征，因此用它覆盖掉duration_info文件
                if col == 'duration_features':
                    with open("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + 'duration' + '_' + "info" + ".npy", 'wb') as file:
                        pickle.dump(att_count, file)

                # 将唯一值个数保存为 Numpy 文件
                with open("raw_dir/" + self._eventlog + "_" + str(self._fold)+ "/" + col + '_' + "info" + ".npy", 'wb') as file:
                    pickle.dump(att_count, file)
                with open('raw_dir/' + self._eventlog + "_" + str(self._fold)+ '/' + 'eventlog' + '_' + 'info' + '.npy', 'wb') as file:
                    pickle.dump(1, file)

            print('外面的\n', train_df)

            self.save_node_feature(train_df, type='train')
            self.save_node_feature(val_df, type='val')
            self.save_node_feature(test_df, type='test')

            # 创建各个数据集的多层异构图
            train_hetro_graph = self.create_hetero_graph(train_df, type='train')
            val_hetro_graph = self.create_hetero_graph(val_df, type='val')
            test_hetro_graph = self.create_hetero_graph(test_df, type='test')


            dgl.save_graphs(f"raw_dir/{self._eventlog}"+ "_" + str(self._fold)+"/Graph/train_hetro_graph", [train_hetro_graph])
            dgl.save_graphs(f"raw_dir/{self._eventlog}"+ "_" + str(self._fold)+"/Graph/val_hetro_graph", [val_hetro_graph])
            dgl.save_graphs(f"raw_dir/{self._eventlog}"+ "_" + str(self._fold)+"/Graph/test_hetro_graph", [test_hetro_graph])














