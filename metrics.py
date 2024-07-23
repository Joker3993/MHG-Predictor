import argparse
import os

import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, \
    precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from tqdm import tqdm

from Dataset import MyDataset

def get_device(gpu):
    """
    根据GPU的可用性获取用于训练的设备。

    参数：
    - gpu：要使用的GPU索引。

    返回：
    - torch.device：要使用的设备（GPU或CPU）。
    """
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')

def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)


def Final_test(args):

    print("start testing...")
    print(f"dataset: {args.dataset}")
    print(f"hidden_dim: {args.hidden_dim}")
    print(f"num_layers: {args.num_layers}")

    print(f"lr: {args.lr}")
    print(f"batch_size: {args.batch_size}")
    print(f"gpu: {args.gpu}")

    # print(f"feat_dropout:{args.feat_dropout}")
    # print(f"dropout: {args.dropout}")

    for fold in range(3):

        result_path = "result_Hetro_homo/" + args.dataset
        # 如果结果保存的路径不存在，就创建该路径
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        # 以追加模式打开结果保存的文件，使用args.eventlog指定文件的名称
        outfile2 = open(result_path + "/" + args.dataset + "_" + ".txt", 'a')


        dataset_test = MyDataset(name=args.dataset +"_" +str(fold), type="test")
        # 创建 dataloader
        test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
        # 指定设备
        device = get_device(args.gpu)

        model_path = 'model_Hetro_homo/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(args.lr) + '_' + str(args.batch_size) + f'_fold{fold}'  + '_model.pkl'

        # """测试参数专用"""
        # model_path = 'model_Test_Hetro/' + str(args.dataset) + '/'+ f'f_drop{str(args.feat_dropout)}'+f'_drop{str(args.dropout)}'+ f'_fold{fold}' + '_model.pkl'

        model = torch.load(model_path)
        model.to(device)

        # 将模型设置为评估模式
        model.eval()
        loss_func = nn.CrossEntropyLoss()

        # 初始化总损失
        total_loss = 0
        total_accuracy = 0
        total_step = 0
        Y_labels = []
        Y_preds = []


        # 在评估模式下，不计算梯度
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次
            for batch in test_loader:
                # 将每个批次的图和标签移到指定的设备上
                hetro_graph, homo_graph, labels = batch
                hetro_graph = hetro_graph.to(device)
                homo_graph = homo_graph.to(device)
                labels = labels.to(device)

                # 前向传播
                logits = model(hetro_graph, homo_graph)

                # 计算损失
                loss = loss_func(logits, labels)

                # 累加当前批次的损失
                total_loss += loss.item()
                total_accuracy += (logits.argmax(1) == labels).sum().item()

                # 计算预测值
                y_pred = logits.argmax(1)
                # 将预测值移回GPU
                y_pred = y_pred.to(device)

                Y_labels.append(labels)
                Y_preds.append(y_pred)

        # 将真实标签列表中的张量拼接起来，并转移到cpu上
        Y_test_int = torch.cat(Y_labels, 0).to('cpu')
        # 将预测标签列表中的张量拼接起来，并转移到cpu上
        preds_a = torch.cat(Y_preds, 0).to('cpu')

        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                       pos_label=None)

        auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")


        print(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: '+ str(prauc_score_macro))
        outfile2.write('\n')

        outfile2.write('\n' )


        outfile2.flush()

        outfile2.close()



if __name__ == '__main__':

    list_eventlog = [

                    'bpi13_closed_problems',
                    # 'bpi12_all_complete',
                    #  'bpi12w_complete',
                    #  'bpi13_incidents',
                    #  'bpi13_problems',
                    #  'bpic2017_o',
                    #  'bpic2020',
                    #  'helpdesk',
                    #  'receipt',
                    #  'bpi12_work_all'
    ]

    for eventlog in tqdm(list_eventlog):

            parser = argparse.ArgumentParser(description='BPIC')

            # 添加命令行参数，用于指定数据集，默认为 "bpi13_problems"
            parser.add_argument("-d", "--dataset", type=str, default=eventlog, help="dataset to use")

            # 添加命令行参数，用于指定隐藏层的维度，默认为 128
            parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")

            # 添加命令行参数，用于指定神经网络的层数，默认为 3
            parser.add_argument("--num-layers", type=int, default=2, help="number of layer")


            # 添加命令行参数，用于指定学习率，默认为 0.001
            parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")

            # 添加命令行参数，用于指定每个训练批次的样本数，默认为 16
            parser.add_argument("--batch-size", type=int, default= 32, help="batch size")

            # 添加命令行参数，用于指定使用的GPU序号，默认为 0
            parser.add_argument("--gpu", type=int, default=3, help="gpu")


            # 解析命令行参数，并将其存储在 args 对象中
            args = parser.parse_args()

            Final_test(args)

