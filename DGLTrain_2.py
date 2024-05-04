import argparse
import copy
import os
import pickle
import random

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from DGLDataset import MyDataset
import warnings

from Hetro_Homo import SAGE_Classifier

warnings.filterwarnings("ignore", category=UserWarning)


class Tran:
    def __init__(self,eventlog):
        self._evenlog = eventlog
        self._fold = 0

    def get_device(self,gpu):
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

    def train(self,model, train_loader, loss_func, optimizer, device, data_length, total_train_step ):
        """
        训练神经网络模型。

        参数：
        - model: 被训练的神经网络模型。
        - train_loader: 用于加载训练数据的数据加载器。
        - criterion: 损失函数，用于计算模型输出与真实标签之间的差异。
        - optimizer: 优化器，用于更新模型的参数以最小化损失。
        - device: 指定模型和数据所在的设备，可以是 CPU 或 GPU。
        - data_length: 训练数据集的大小，用于计算平均损失和准确率。

        返回：
        - 平均损失
        - 平均准确率
        """

        # 将模型设置为训练模式
        model.train()

        # 初始化总损失和总准确率
        total_loss = 0

        total_accuracy = 0
        total_step = 0
         # 遍历训练数据加载器中的每个批次
        for batch in train_loader:

            # 将每个批次的图和标签移到指定的设备上
            hetro_graph, homo_graph, labels = batch
            hetro_graph = hetro_graph.to(device)
            homo_graph = homo_graph.to(device)
            labels = labels.to(device)
            # print(f'labels:{labels}')

            # 前向传播
            logits = model(hetro_graph,homo_graph)
            logits = logits.to(device)
            # print(f'logits:{logits}')


            # 计算损失
            loss = loss_func(logits ,labels)


            # 反向传播和参数更新.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加当前批次的损失
            total_loss += loss.item()

            total_accuracy += (logits.argmax(1) == labels).sum().item()


            # #每批次的训练loss都打印太繁琐，每50次打印一下loss
            # total_train_step +=BPI_Challenge_2012_W
            # if total_train_step % 50 == 0:
            #     print(f"训练次数：{total_train_step},Loss:{loss.item()}")

            # # # """看下标签和预测值,loss"""
            # print(logits.squeeze())
            # print(labels)
            # print('loss',loss)

            total_step += 1

        # 返回结果
        return  total_loss / total_step , total_accuracy / data_length


    def validate(self,model,loss_func,val_loader, device, data_length):
        """
        在验证集上评估模型的准确率。

        参数：
        - model: 被评估的神经网络模型。
        - val_loader: 用于加载验证数据的数据加载器。
        - device: 指定模型和数据所在的设备，可以是 CPU 或 GPU。
        - data_length: 验证数据集的大小，用于计算平均准确率。

        返回：
        - 平均准确率
        """

        # 将模型设置为评估模式
        model.eval()

        # 初始化总损失
        total_loss = 0
        total_accuracy = 0
        total_test_step = 0

        # 在评估模式下，不计算梯度
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次
            for batch in val_loader:

                # 将每个批次的图和标签移到指定的设备上
                hetro_graph, homo_graph, labels = batch
                hetro_graph = hetro_graph.to(device)
                homo_graph = homo_graph.to(device)
                labels = labels.to(device)

                # 前向传播
                logits = model(hetro_graph,homo_graph)
                logits = logits.to(device)

                # 计算损失
                loss = loss_func(logits, labels)

                # 累加当前批次的损失
                total_loss += loss.item()
                total_accuracy += (logits.argmax(1) == labels).sum().item()
                total_test_step += 1

        return total_loss / total_test_step, total_accuracy / data_length



    def test(self,model,loss_func,val_loader, device, data_length):
        """
        在验证集上评估模型的准确率。

        参数：
        - model: 被评估的神经网络模型。
        - val_loader: 用于加载验证数据的数据加载器。
        - device: 指定模型和数据所在的设备，可以是 CPU 或 GPU。
        - data_length: 验证数据集的大小，用于计算平均准确率。

        返回：
        - 平均准确率
        """

        # 将模型设置为评估模式
        model.eval()

        # 初始化总损失
        total_loss = 0
        total_accuracy = 0

        total_step = 0


        # 在评估模式下，不计算梯度
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次
            for batch in val_loader:
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

                total_step += 1


        # 计算平均损失
        average_loss = total_loss / total_step
        average_accuracy = total_accuracy / data_length

        # 返回结果
        return average_loss , average_accuracy


    def train_val(self,args):
        """
        训练和验证神经网络模型。

        参数：
        - args: 命令行参数，包含训练和模型配置信息。

        该函数执行以下步骤：
        BPI_Challenge_2012_W. 打印训练配置信息。
        2. 加载训练、验证和测试数据集。
        3. 初始化神经网络模型。
        4. 创建数据加载器。
        5. 将模型移至指定设备。
        6. 定义损失函数和优化器。
        7. 进行训练循环，并在每个 epoch 输出训练损失和准确率，以及验证准确率。
        8. 保存在验证集上表现最好的模型。
        9. 打印最佳 epoch 的验证准确率和在测试集上的准确率。

        返回：
        无
        """

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
        print(f"num_heads: {args.num_heads}")
        print(f"weight_decay:{args.weight_decay}")

        for fold in range(3):

            np.random.seed(10)
            random.seed(10)
            torch.manual_seed(10)  # 设置 PyTorch 的随机种子

            self._fold = fold

            print(f"--------------------------------------第{self._fold}折开始-------------------------------------------")

            # 数据导入
            dataset_train = MyDataset(name=args.dataset +"_" +str(self._fold), type="train")
            dataset_val = MyDataset(name=args.dataset +"_" +str(self._fold), type="val")
            dataset_test = MyDataset(name=args.dataset +"_" +str(self._fold), type="test")

            node_name_path = "raw_dir/" + args.dataset + "_" + str(self._fold) + "/" + "node_name" + ".npy"


            with open( "raw_dir/" + args.dataset +"_" + str(self._fold) + "/" +  "etypes" + ".npy", 'rb') as file:
                etypes = pickle.load(file)
            rel_name = etypes

            # 指定设备
            device = self.get_device(args.gpu)

            with open(node_name_path, 'rb') as file:
                node_name = pickle.load(file)

            vocab_sizes = [np.load("raw_dir/" + args.dataset +"_" + str(self._fold) + "/" + node_name + "_info.npy", allow_pickle=True) for node_name in
                           node_name]

            model = SAGE_Classifier(args.hidden_dim,args.n_classes,args.num_heads,args.num_layers,node_name=node_name,vocab_sizes=vocab_sizes,rel_names=rel_name)

            # 创建 dataloaders
            train_loader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
            val_loader = GraphDataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
            test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)


            # """测试dataloader的代码"""
            # for batch in train_loader:
            #     hetro_graph, homo_graph, label = batch
            #     print("Heterogeneous Graph Structure:")
            #     print(hetro_graph.nodes['duration'].data['duration'])
            #     print(label)
            #
            #     print("Homogeneous Graph Structure:")
            #     print(homo_graph)
            #     print("Label:")
            #     print(label)
            #     # 可以添加其他需要查看的信息
            #     print("=" * 50)


            # 将模型移至指定设备

            model.to(device)

            # 定义损失函数和优化器

            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


            patience = 10
            no_improvement_count = 0
            best_epoch = 0
            best_model = None
            total_train_step = 0

            best_val_acc = 0
            train_loss_list = []
            validation_loss_list = []

            # 训练循环
            for epoch in range(args.num_epochs):
                print(f"------第{epoch+1}轮训练开始-----")

                train_loss,train_accuracy = self.train(model, train_loader, loss_func, optimizer, device, len(dataset_train),total_train_step)
                # print(f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss:.5f}, Training Accuracy: {train_accuracy:.5f}')

                val_loss ,val_accuracy= self.validate(model, loss_func,val_loader, device, len(dataset_val))
                print(f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss:.3f},Validation Loss: {val_loss:.3f},, Training Accuracy: {train_accuracy:.3f}, Validation Accuracy: {val_accuracy:.3f}')

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

            # # 保存最佳模型
            # path = "model_temp/" + args.dataset
            # if not os.path.exists(path):
            #     os.makedirs(path,exist_ok=True)
            # # model_path = 'model/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(args.lr) + '_model.pkl'
            # model_path = 'model_temp/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(args.lr) +'_'+ str(args.batch_size)+ f'_fold{self._fold}' + '_model.pkl'

            # 保存最佳模型
            path = "model_Hetro_homo/" + args.dataset
            if not os.path.exists(path):
                os.makedirs(path,exist_ok=True)
            # model_path = 'model/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(args.lr) + '_model.pkl'
            model_path = 'model_Hetro_homo/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(args.lr) +'_'+ str(args.batch_size)+ f'_fold{self._fold}' + '_model.pkl'

            torch.save(best_model, model_path)

            # 验证最佳模型在验证集和测试集上的准确率
            check_model = torch.load(model_path)
            val_loss,val_accuracy = self.validate(check_model, loss_func,val_loader, device, len(dataset_val))
            print('-' * 89)
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Validation Loss: {val_loss:.5f}')

            print('-' * 89)
            test_loss, test_accuracy= self.test(check_model,loss_func, test_loader, device, len(dataset_test))
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test Loss: {test_loss:.5f}')
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test average Accuracy:{test_accuracy:.5f}')

            print('-' * 89)
            print('Training finished.')




    def tran_main(self):


        class_num = np.load("raw_dir/" + self._evenlog + "_" + str(self._fold) + "/" + "activity" + "_info.npy",allow_pickle=True)

        # 创建一个 ArgumentParser 对象，用于处理命令行参数
        parser = argparse.ArgumentParser(description='BPIC')

        # 添加命令行参数，用于指定数据集，默认为 "bpi13_problems"
        parser.add_argument("-d", "--dataset", type=str, default=self._evenlog, help="dataset to use")

        # 添加命令行参数，用于指定隐藏层的维度，默认为 128
        parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")

        # 添加命令行参数，用于指定神经网络的层数，默认为 3
        parser.add_argument("--num-layers", type=int, default=2, help="number of layer")

        # 添加命令行参数，用于指定分类的数量，默认为 36
        parser.add_argument("--n-classes", type=int, default= class_num, help="number of class")

        """bpi13_problems: 7,bpic2017_o: 8 , receipt：27 , bpi13_incidents：13, bpi12w_complete :6 ,bpi12_work_all: 19, bpi12_all_complete: 23 ,bpic2020: 19,
            helpdesk : 14 ,"""

        # 添加命令行参数，用于指定训练的轮数，默认为 20
        parser.add_argument("--num-epochs", type=int, default=30, help="number of epoch")

        # 添加命令行参数，用于指定学习率，默认为 0.001
        parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")

        # 添加命令行参数，用于指定每个训练批次的样本数，默认为 16
        parser.add_argument("--batch-size", type=int, default= 32, help="batch size")

        # 添加命令行参数，用于指定使用的GPU序号，默认为 0
        parser.add_argument("--gpu", type=int, default=0, help="gpu")

        parser.add_argument("--num_heads", type=int, default=8, help="num_heads")

        parser.add_argument("--weight_decay", type=int, default=0, help="weight_decay")
        # 解析命令行参数，并将其存储在 args 对象中
        args = parser.parse_args()

        # 调用 train_val 函数，并传入解析后的命令行参数 args 进行模型训练
        self.train_val(args)


