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
    for fold in range(3):
        result_path = "result_Hetro_homo/" + args.dataset
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        outfile2 = open(result_path + "/" + args.dataset + "_" + ".txt", 'a')
        dataset_test = MyDataset(name=args.dataset + "_" + str(fold), type="test")
        test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
        device = get_device(args.gpu)
        model_path = 'model_Hetro_homo/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(
            args.num_layers) + '_' + str(args.lr) + '_' + str(args.batch_size) + f'_fold{fold}' + '_model.pkl'
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        loss_func = nn.CrossEntropyLoss()
        total_loss = 0
        total_accuracy = 0
        total_step = 0
        Y_labels = []
        Y_preds = []
        with torch.no_grad():
            for batch in test_loader:
                hetro_graph, homo_graph, labels = batch
                hetro_graph = hetro_graph.to(device)
                homo_graph = homo_graph.to(device)
                labels = labels.to(device)
                logits = model(hetro_graph, homo_graph)
                loss = loss_func(logits, labels)
                total_loss += loss.item()
                total_accuracy += (logits.argmax(1) == labels).sum().item()
                y_pred = logits.argmax(1)
                y_pred = y_pred.to(device)
                Y_labels.append(labels)
                Y_preds.append(y_pred)
        Y_test_int = torch.cat(Y_labels, 0).to('cpu')
        preds_a = torch.cat(Y_preds, 0).to('cpu')
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                       pos_label=None)
        auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")
        print(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')
        outfile2.write('\n')
        outfile2.flush()
        outfile2.close()


if __name__ == '__main__':
    list_eventlog = [
        # 'bpi13_closed_problems',
        # 'bpi12w_complete',
        # 'bpi13_problems',
        'bpic2017_o',
        # 'helpdesk',
        # 'bpi12_work_all'
    ]
    for eventlog in tqdm(list_eventlog):
        parser = argparse.ArgumentParser(description='BPIC')
        parser.add_argument("-d", "--dataset", type=str, default=eventlog, help="dataset to use")
        parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")
        parser.add_argument("--num-layers", type=int, default=2, help="number of layer")
        parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
        parser.add_argument("--batch-size", type=int, default=32, help="batch size")
        parser.add_argument("--gpu", type=int, default=0, help="gpu")
        args = parser.parse_args()
        Final_test(args)
