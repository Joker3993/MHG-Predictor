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


# 定义一个函数，它接受一个数据框作为参数，返回两个列表，一个是活动序列，一个是案例编号
class SubGraph:
  def __init__(self,eventlog):
    self._eventlog = eventlog
    self._fold = 0

  def get_sequences_activity(self,df):
    # 创建一个空的列表，用来存储活动序列
    sequences = []
    # 创建一个空的列表，用来存储案例编号
    cases = []
    # 按照案例对数据框进行分组
    groups = df.groupby('case')

    # 遍历每个分组
    for case, group in groups:
      # 获取该分组的活动列，转换为列表
      activities = group['node_col'].tolist()

      # """处理只有一个活动的特殊情况，失败了，会出现新问题"""
      # if len(activities) == BPI_Challenge_2012_W:#特殊情况
      #   sequences.append([])
      #   cases.append(case)

      # 遍历该列表的每个元素，从第一个元素开始
      for i in range(len(activities)-1):

        # 获取该元素之前的所有元素（包括自身），形成一个子列表，这就是一个前缀序列
        prefix = activities[:i+1]
        # 将这个前缀序列添加到序列列表中
        sequences.append(prefix)
        # 将该分组的案例编号添加到案例编号列表中
        cases.append(case)
    # 返回两个列表
    return sequences, cases


  # 定义一个函数，它接受一个数据框作为参数，返回三个列表，一个是node_col序列，一个是案例编号，一个是duration标签
  def get_sequences_time(self,df):
    # 创建一个空的列表，用来存储node_col序列
    sequences = []
    # 创建一个空的列表，用来存储案例编号
    cases = []
    # 创建一个空的列表，用来存储duration标签
    labels = []
    # 按照案例对数据框进行分组
    groups = df.groupby('case')
    # 遍历每个分组
    for case, group in groups:
      # 获取该分组的node_col列，转换为列表
      node_cols = group['node_col'].tolist()
      # 获取该分组的duration列，转换为列表
      durations = group['activity'].tolist()

      # """处理只有一个活动的特殊情况，失败了，会出现新问题"""
      # if len(node_cols) == BPI_Challenge_2012_W:
      #   sequences.append([])
      #   cases.append(case)
      #   labels.append(durations[0])

      # 遍历node_col列表的每个元素，从第一个元素开始，到倒数第二个元素结束
      for i in range(len(node_cols)-1):
        # 获取该元素之前的所有元素（包括自身），形成一个子列表，这就是一个前缀序列
        prefix = node_cols[:i+1]
        # 将这个前缀序列添加到序列列表中
        sequences.append(prefix)
        # 将该分组的案例编号添加到案例编号列表中
        cases.append(case)

        # 获取duration列表中对应的下一个元素，这就是一个标签
        label = durations[i+1]
        # 将这个标签添加到标签列表中
        labels.append(label)

    # 返回三个列表
    return sequences, cases, labels


  def subgraph(self,data,loaded_graphs,type):
    data['node_col'] = range(len(data))
    """案例列需要再次整数编码"""
    col = 'case'
    # 使用encode_map对该列进行编码
    att_encode_map = encode_map(set(data[col].values))
    # 利用上面生成的字典，将各列对应字符转换为整数
    data[col] = data[col].apply(lambda e: att_encode_map.get(str(e), -1))

    # """测试样例"""
    # df={'case':[BPI_Challenge_2012_W,BPI_Challenge_2012_W,2,2,2,3,0,0,4],
    #     'node_col':[0,BPI_Challenge_2012_W,2,3,4,5,6,7,8],
    #     'duration':[10,11,12,13,14,15,16,17,18]}
    # df=pd.DataFrame(df)

    # 调用函数，传入数据框，得到两个列表.seq_case1和seq_case2其实是一样的。
    seq_activity, seq_case1 = self.get_sequences_activity(data)
    seq_duration, seq_case2, seq_labels = self.get_sequences_time(data)

    # print('活动前缀\n',seq_activity)
    # print('案例号\n',seq_case1)
    # print("标签\n",seq_labels)
    # print('持续时间前缀\n',seq_duration)

    #把活动序列、案例数据、标签数据、持续时间保存成二进制文件
    with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_activity" + ".npy", 'wb') as file:
      pickle.dump(seq_activity, file)
    with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_case1" + ".npy", 'wb') as file:
      pickle.dump(seq_case1, file)
    with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_labels" + ".npy", 'wb') as file:
      pickle.dump(seq_labels, file)
    with open("raw_dir/" + self._eventlog + "_" + str(self._fold) + "/" + type + '_' + "seq_duration" + ".npy", 'wb') as file:
      pickle.dump(seq_duration, file)


    list_hetro_subgraph = []
    list_homo_subgraph = []

    """异构子图"""
    # for i in tqdm(range(len(seq_case1))):
    #   hetro_subgraph = dgl.node_subgraph(loaded_graphs[0], {'eventlog': [0],
    #                                                         'case': seq_case1[i],
    #                                                         "duration": seq_duration[i],
    #                                                         "activity": seq_activity[i],
    #
    #                                                         "resource": seq_activity[i] ,
    #                                                         # 'Amount': seq_activity[i],
    #                                                          'Att1': seq_activity[i],'Att2': seq_activity[i],'Att3': seq_activity[i],'Att4': seq_activity[i],
    #                                                          'Att5': seq_activity[i],
    #                                                         'Att6': seq_activity[i],
    #                                                         # 'Att7': seq_activity[i],
    #                                                         # 'Att8': seq_activity[i],'Att9': seq_activity[i]
    #
    #                                                         })
    # print("111",loaded_graphs[0].ntypes)

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
      # print(node_dict)

      hetro_subgraph = dgl.node_subgraph(loaded_graphs[0], nodes=node_dict)
      list_hetro_subgraph.append(hetro_subgraph)

      # print(f'第{i}张异构子图\n', hetro_subgraph)
      # print(f'第{i}张异构子图\n', hetro_subgraph.nodes['duration'].data['duration'])

    """同构子图"""
    for i in tqdm(range(len(seq_case1))):
      homo_subgraph = dgl.node_subgraph(loaded_graphs[0], {"activity":seq_activity[i]})
      # list_homo_subgraph.append(homo_subgraph)
      if homo_subgraph.num_edges('next') == 0:
        homo_subgraph =dgl.add_self_loop(homo_subgraph,etype='next')
      # print(f'第{i}张同构子图\n', homo_subgraph)
      # print(f'第{i}张同构子图\n',homo_subgraph.ndata['activity'])
      list_homo_subgraph.append(dgl.to_homogeneous(homo_subgraph, ndata=['activity']))


    return list_hetro_subgraph,list_homo_subgraph


  def Subgraph_main(self):

    for fold in range(3):
      self._fold = fold

      print(f"--------------------------------------{self._eventlog}第{self._fold}折开始-------------------------------------------")

      for type in ['train','test','val']:
        """加载整图"""
        load_path = "raw_dir/"+ self._eventlog + "_" + str(self._fold) + f"/Graph/{type}_hetro_graph"
        loaded_graphs,_ = dgl.load_graphs(load_path)#加载了一个存储着图的图列表
        # print(loaded_graphs[0])

        path = "./raw_dir" + "/" + self._eventlog + "_" + str(self._fold)
        features_path = os.path.join(path, "feature_list.npy")
        feature_lists = np.load(features_path, allow_pickle=True)
        feature_lists.append('case')
        feature_lists.append('duration')

        node_features = {}

        for feature_name in feature_lists:
          if feature_name == 'duration':
            #标签还是要用原始数据，特征改为聚类后的类别值
            feature_path = os.path.join(path, f"{type}_{feature_name}_labels.npy")
            att_lists = np.load(feature_path, allow_pickle=True)
            node_features[feature_name] = att_lists
          else:
            feature_path = os.path.join(path, f"{type}_{feature_name}.npy")
            att_lists = np.load(feature_path, allow_pickle=True)
            node_features[feature_name] = att_lists


        data = pd.DataFrame(node_features)
        train_hetro_subgraph,train_homo_subgraph = self.subgraph(data,loaded_graphs,type)

        print(data)

        dgl.save_graphs("raw_dir/"+ self._eventlog + "_" + str(self._fold) +f"/Graph/{type}_hetro_subgraph", train_hetro_subgraph)
        dgl.save_graphs("raw_dir/"+ self._eventlog + "_" + str(self._fold) +f"/Graph/{type}_homo_subgraph", train_homo_subgraph)

        print(self._eventlog + "_" + str(self._fold) +"_" +f'{type}数据集子图生成完毕\n')





