import argparse
import collections
import copy
import os
import pickle
import random
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.utils import compute_class_weight, compute_sample_weight
from torch import nn, optim
import matplotlib.pyplot as plt
from Dataset import MyDataset
import warnings
from Hetro_Homo import SAGE_Classifier
warnings.filterwarnings("ignore", category=UserWarning)
class Tran:
    def __init__(self, eventlog):
        self._evenlog = eventlog
        self._fold = 0
    def get_device(self, gpu):
        if torch.cuda.is_available() and gpu < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu}')
        else:
            return torch.device('cpu')
    def train(self, model, train_loader, loss_func, optimizer, device, data_length, total_train_step):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_step = 0
        for batch in train_loader:
            hetro_graph, homo_graph, labels = batch
            hetro_graph = hetro_graph.to(device)
            homo_graph = homo_graph.to(device)
            labels = labels.to(device)
            logits = model(hetro_graph, homo_graph)
            logits = logits.to(device)
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()
            total_step += 1
        return total_loss / total_step, total_accuracy / data_length
    def validate(self, model, loss_func, val_loader, device, data_length):
        model.eval()
        total_loss = 0
        total_accuracy = 0
        total_test_step = 0
        with torch.no_grad():
            for batch in val_loader:
                hetro_graph, homo_graph, labels = batch
                hetro_graph = hetro_graph.to(device)
                homo_graph = homo_graph.to(device)
                labels = labels.to(device)
                logits = model(hetro_graph, homo_graph)
                logits = logits.to(device)
                loss = loss_func(logits, labels)
                total_loss += loss.item()
                total_accuracy += (logits.argmax(1) == labels).sum().item()
                total_test_step += 1
        return total_loss / total_test_step, total_accuracy / data_length
    def test(self, model, loss_func, val_loader, device, data_length):
        model.eval()
        total_loss = 0
        total_accuracy = 0
        total_step = 0
        with torch.no_grad():
            for batch in val_loader:
                hetro_graph, homo_graph, labels = batch
                hetro_graph = hetro_graph.to(device)
                homo_graph = homo_graph.to(device)
                labels = labels.to(device)
                logits = model(hetro_graph, homo_graph)
                loss = loss_func(logits, labels)
                total_loss += loss.item()
                total_accuracy += (logits.argmax(1) == labels).sum().item()
                total_step += 1
        average_loss = total_loss / total_step
        average_accuracy = total_accuracy / data_length
        return average_loss, average_accuracy
    def train_val(self, args):
        print("start training...")
        print("Training with the following arguments:")
        print(f"dataset: {args.dataset}")
        print(f"hidden_dim: {args.hidden_dim}")
        print(f"num_layers: {args.num_layers}")
        print(f"n_classes: {args.n_classes}")
        print(f"num_epochs: {args.num_epochs}")
        print(f"lr: {args.lr}")
        print(f"batch_size: {args.batch_size}")
        print(f"gpu: {args.gpu}")
        for fold in range(3):
            self._fold = fold
            print(
                f"--------------------------------------第{self._fold}折开始-------------------------------------------")
            dataset_train = MyDataset(name=args.dataset + "_" + str(self._fold), type="train")
            dataset_val = MyDataset(name=args.dataset + "_" + str(self._fold), type="val")
            dataset_test = MyDataset(name=args.dataset + "_" + str(self._fold), type="test")
            node_name_path = "raw_dir/" + args.dataset + "_" + str(self._fold) + "/" + "node_name" + ".npy"
            with open("raw_dir/" + args.dataset + "_" + str(self._fold) + "/" + "etypes" + ".npy", 'rb') as file:
                etypes = pickle.load(file)
            rel_name = etypes
            device = self.get_device(args.gpu)
            with open(node_name_path, 'rb') as file:
                node_name = pickle.load(file)
            vocab_sizes = [np.load("raw_dir/" + args.dataset + "_" + str(self._fold) + "/" + node_name + "_info.npy",
                                   allow_pickle=True) for node_name in node_name]
            model = SAGE_Classifier(args.hidden_dim, args.n_classes, args.num_layers, node_name=node_name,vocab_sizes=vocab_sizes, rel_names=rel_name)
            train_loader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
            val_loader = GraphDataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
            test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
            model.to(device)
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.NAdam(model.parameters(), lr=args.lr)
            patience = 10
            no_improvement_count = 0
            best_epoch = 0
            best_model = None
            total_train_step = 0
            best_val_acc = 0
            train_loss_list = []
            validation_loss_list = []
            for epoch in range(args.num_epochs):
                print(f"------第{epoch + 1}轮训练开始-----")
                train_loss, train_accuracy = self.train(model, train_loader, loss_func, optimizer, device,
                                                        len(dataset_train), total_train_step)
                val_loss, val_accuracy = self.validate(model, loss_func, val_loader, device, len(dataset_val))
                print(
                    f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss:.3f},Validation Loss: {val_loss:.3f},, Training Accuracy: {train_accuracy:.3f}, Validation Accuracy: {val_accuracy:.3f}')
                train_loss_list.append(train_loss)
                validation_loss_list.append(val_loss)
                if val_accuracy >= best_val_acc:
                    best_val_acc = val_accuracy
                    best_epoch = epoch + 1
                    no_improvement_count = 0
                    best_model = copy.deepcopy(model)
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("Early stopping: Validation accuracy has not improved for {} epochs.".format(patience))
                        break
            path = "model_Hetro_homo/" + args.dataset
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            model_path = 'model_Hetro_homo/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(
                args.num_layers) + '_' + str(args.lr) + '_' + str(args.batch_size) + f'_fold{self._fold}' + '_model.pkl'
            torch.save(best_model, model_path)
            check_model = torch.load(model_path)
            val_loss, val_accuracy = self.validate(check_model, loss_func, val_loader, device, len(dataset_val))
            print('-' * 89)
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Validation Loss: {val_loss:.5f}')
            print('-' * 89)
            test_loss, test_accuracy = self.test(check_model, loss_func, test_loader, device, len(dataset_test))
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test Loss: {test_loss:.5f}')
            print(
                f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test average Accuracy:{test_accuracy:.5f}')
            print('-' * 89)
            print('Training finished.')
    def tran_main(self):
        class_num = np.load("raw_dir/" + self._evenlog + "_" + str(self._fold) + "/" + "activity" + "_info.npy",
                            allow_pickle=True)
        parser = argparse.ArgumentParser(description='BPIC')
        parser.add_argument("-d", "--dataset", type=str, default=self._evenlog, help="dataset to use")
        parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")
        parser.add_argument("--num-layers", type=int, default=2, help="number of layer")
        parser.add_argument("--n-classes", type=int, default=class_num, help="number of class")
        parser.add_argument("--num-epochs", type=int, default=100, help="number of epoch")
        parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
        parser.add_argument("--batch-size", type=int, default=32, help="batch size")
        parser.add_argument("--gpu", type=int, default=0, help="gpu")
        args = parser.parse_args()
        self.train_val(args)
