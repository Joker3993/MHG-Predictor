import numpy as np
import torch

np.random.seed(133)


def encode_map(input_array):
    p_map = {}
    length = len(input_array)
    for index, ele in zip(range(length), input_array):
        p_map[str(ele)] = index
    return p_map


def decode_map(encode_map):
    de_map = {}
    for k, v in encode_map.items():
        de_map[v] = k
    return de_map


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

    while i < len(sequence):
        if len(sequence.iat[i, 0]) == 1:
            src = sequence.iat[i, 0]
            dst = sequence.iat[i, 0]
        else:
            src = sequence.iat[i, 0][:-1]
            dst = sequence.iat[i, 0][1:]
        activity_list_src.append(src)
        activity_list_dst.append(dst)
        case_list.append(i)
        i = i + 1
    return activity_list_src, activity_list_dst, case_list
