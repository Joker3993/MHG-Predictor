import numpy as np
import torch
# 编码方法
np.random.seed(133)
def encode_map(input_array):
    p_map={}
    length=len(input_array)
    for index, ele in zip(range(length),input_array):
        # print(ele,index)
        p_map[str(ele)] = index
    return p_map


# 解码方法
def decode_map(encode_map):
    de_map={}
    for k,v in encode_map.items():
        # index,ele
        de_map[v]=k
    return de_map

"""获取前缀序列的函数"""
def get_prefix_sequence(sequence):
    i = 0
    list_seq = []
    while i < len(sequence):
        list_temp = []
        j = 0
        while j < (len(sequence.iat[i, 0]) - 1):
            list_temp.append(sequence.iat[i, 0][0 + j])
            list_seq.append(list(list_temp))
            j = j + 1
        i = i + 1
    return list_seq

"""函数功能：
在处理序列时，函数将当前元素作为前缀序列的一部分，将下一个元素作为相应的标签。
提取的前缀序列、标签以及案例的索引 存储在三个列表中：list_seq、list_label 和 list_case"""
def get_prefix_sequence_label(sequence):

    i = 0
    list_seq = []
    list_label = []
    list_case = []
    while i < len(sequence):
        list_temp = []
        j = 0
        while j < (len(sequence.iat[i, 0]) - 1):
            list_temp.append(sequence.iat[i, 0][0 + j])
            list_seq.append(list(list_temp))
            list_label.append(sequence.iat[i, 0][j + 1])
            list_case.append(i)
            j = j + 1
        i = i + 1
    return list_seq, list_label, list_case

def compute_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def create_activity_activity(sequence):

    i = 0
    activity_list_src = []
    activity_list_dst = []
    case_list = []

    # 遍历每个案例
    while i < len(sequence):
        if len(sequence.iat[i,0]) == 1:
            src = sequence.iat[i,0]
            dst = sequence.iat[i,0]
        else:
            src = sequence.iat[i,0][:-1]  # 边的起始节点
            dst = sequence.iat[i,0][1:]  # 边的终止节点

        activity_list_src.append(src)
        activity_list_dst.append(dst)
        case_list.append(i)
        i = i + 1

    return activity_list_src, activity_list_dst,case_list