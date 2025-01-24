# -*- coding: utf-8 -*-
from datetime import datetime
import time 
import argparse
import copy
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,cohen_kappa_score
import os
import sys
import random
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
import torch.utils.data as Data
from model.grid_cross5_model_S2_deng import model_S0

import pickle
import dgl
import csv
#from data_preprocessing import create_graph_data,drug_to_mol_graph,BipartiteData,get_bipartite_graph
from torch_geometric.data import Batch
from gridsearch_deng_S2_data_helper import construct_graph, get_kg

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

zhongzi = 1
torch.manual_seed(zhongzi)
torch.cuda.manual_seed(zhongzi)
torch.cuda.manual_seed_all(zhongzi)
random.seed(zhongzi)
np.random.seed(zhongzi)
torch.backends.cudnn.deterministic = True  # 保证每次训练的结果一致
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def do_compute_metrics(probas_pred, target):
    y_pred_train1 = []
    y_label_train = np.array(target)
    y_label_train=y_label_train.reshape((-1))
    y_pred_train = np.array(probas_pred).reshape((-1, 65))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break

    y_pred_train1=np.array(y_pred_train1).reshape((-1))
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    kappa = cohen_kappa_score(y_label_train, y_pred_train1)
    aaa=y_pred_train1
    bbb=y_label_train

    return acc, f1_score1, recall1,precision1,kappa,aaa,bbb


def generate_drugfeatures(drug1s,drug2s,morgen,alignn,mol):
    drug1_morgen = morgen[drug1s.long()]
    drug2_morgen = morgen[drug2s.long()]
    drug1_alignn = alignn[drug1s.long()]
    drug2_alignn = alignn[drug2s.long()]
    drug1_mol = mol[drug1s.long()]
    drug2_mol = mol[drug2s.long()]

    return drug1_morgen,drug2_morgen,drug1_alignn,drug2_alignn,drug1_mol,drug2_mol

def train(model, train_data_loader, index,l,dim,c, test_data_loader,loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())

    src, dst, rel, hr2eid, rt2eid = construct_graph(index)
    graph = get_kg(src, dst, rel, device)

    best_f1 = -1
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        val_loss = 0

        train_probas_pred = []
        train_ground_truth = []

        model.train()
        for batch in train_data_loader:

            drug1s, drug2s,labels= batch

            outs,_=model(drug1s, drug2s,graph,drug1s, drug2s)

            loss = loss_fn(outs, labels)

            outs=outs.detach().cpu().numpy()
            labels=labels.detach().cpu().numpy()
            train_probas_pred.append(np.array(outs))
            train_ground_truth.append(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(outs)

        train_loss /= len(train_ground_truth)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_f1, train_recall,train_precision,kappa,_,_ = do_compute_metrics(train_probas_pred, train_ground_truth)

            # if scheduler:
            scheduler.step()


            test_probas_pred=[]
            test_ground_truth=[]
            val_drug1s = []
            val_drug2s = []
            for batch in test_data_loader:

                drug1s, drug2s, labels,simdrug1s,simdrug2s  = batch
                outs,_= model(drug1s, drug2s, graph,simdrug1s,simdrug2s)

                probas_pred = outs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                test_probas_pred.append(np.array(probas_pred))
                test_ground_truth.append(labels)
                val_drug1s.append(drug1s.detach().cpu().numpy())
                val_drug2s.append(drug2s.detach().cpu().numpy())

            test_probas_pred = np.concatenate(test_probas_pred)
            test_ground_truth = np.concatenate(test_ground_truth)
            val_drug1s = np.concatenate(val_drug1s).reshape((-1))
            val_drug2s = np.concatenate(val_drug2s).reshape((-1))
            test_acc, test_f1, test_recall, test_precision, test_kappa, predlabls,truelabels = do_compute_metrics(test_probas_pred,test_ground_truth)
            if best_f1 < train_f1:
                best_f1 = train_f1
                torch.save(model.state_dict(), 'S2_MGFF_DDIMDL_cross' + str(dim)+str(l)+str(c)+str(index) + '.pkl')
                print('model saved')
                predlabls += 1
                truelabels += 1
                concatenated = np.concatenate(
                    (val_drug1s[:, np.newaxis], val_drug2s[:, np.newaxis], truelabels[:, np.newaxis],
                     predlabls[:, np.newaxis]),
                    axis=1)
                # 保存到CSV文件
                #np.savetxt('MGFF_S2_label_results' + str(index) + '.csv', concatenated, delimiter=',')
                # 将每一折的结果保存到同一个CSV文件中
                #with open('S2_MGFF_gridvalid_nomol_results.csv', 'a') as f:
                    #f.write(f"{index},{l},{dim},{c},{test_acc}, {test_f1}, {test_recall}, {test_precision}, {test_kappa}\n")

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')
        print("test:", test_acc, test_f1, test_recall, test_precision, test_kappa)


######################### Parameters ######################
parser = argparse.ArgumentParser(description="Parser for DDI")

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=20, help='num of epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--zhongzi', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=128)#256

if __name__ == '__main__':
    args = parser.parse_args()

    def run_model(l,dim,c):#params

        weight_decay = args.weight_decay
        lr = args.lr#args.b#params['lr']
        batch_size = args.batchsize#params['n_batch']
        n_epochs=args.n_epochs


        for index in range(1,6):
        #for index in range(5, 0, -1):
            ###### Dataset
            df_ddi_train = pd.read_csv('../data/Dengcross5datasets/Deng_S2/S2_DDIMDL_idx_train_cross_'+str(index)+'.csv',header=None)
            train_drug1 = torch.tensor(np.array(df_ddi_train.values[:, 0], dtype=int)).to(device=device)#1:1000
            train_drug2 = torch.tensor(np.array(df_ddi_train.values[:, 1], dtype=int)).to(device=device)
            train_label = torch.tensor(np.array(df_ddi_train.values[:, 2], dtype=int)).to(device=device)
            train_label = train_label - 1
            traindata = Data.TensorDataset(train_drug1, train_drug2, train_label)

            df_ddi_test = pd.read_csv('../data/Dengcross5datasets/Deng_S2/S2_DDIMDL_idx_test_similar_cross_'+str(index)+'.csv',header=None)
            test_drug1 = torch.tensor(np.array(df_ddi_test.values[:, 0], dtype=int)).to(device=device)
            test_drug2 = torch.tensor(np.array(df_ddi_test.values[:, 1], dtype=int)).to(device=device)
            test_label = torch.tensor(np.array(df_ddi_test.values[:, 2], dtype=int)).to(device=device)
            test_simdrug1 = torch.tensor(np.array(df_ddi_test.values[:, 3], dtype=int)).to(device=device)
            test_simdrug2 = torch.tensor(np.array(df_ddi_test.values[:, 4], dtype=int)).to(device=device)
            test_label = test_label - 1
            testdata = Data.TensorDataset(test_drug1, test_drug2, test_label,test_simdrug1,test_simdrug2)

            print(f"Training with {len(traindata)} samples, validating with {len(testdata)}")

            train_data_loader = Data.DataLoader(traindata, batch_size=batch_size, shuffle=True, drop_last=True)
            test_data_loader = Data.DataLoader(testdata, batch_size=batch_size, drop_last=False)

            model = model_S0(device,index,dim,l,c).to(device=device)
            # if index !=5:
            #     preindex=index+1
            #     model.load_state_dict(torch.load('S2align_model86_cross' + str(preindex) + '.pkl'))
            loss=torch.nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)#
            #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            print("train!",index,dim,l,c)

            train(model, train_data_loader, index,l,dim,c,test_data_loader, loss, optimizer, n_epochs, device,scheduler)
    # 设置超参数网格
    ls = [1, 2, 3, 4, 5]
    dims = [32, 64, 128, 256, 512]
    cs = [1, 2, 3, 4]

    # for l in ls:
    #     for dim in dims:
    #         # 根据 dim 限制 c 的范围
    #         if dim == 32:
    #             restricted_cs = [1, 2]
    #         elif dim == 64:
    #             restricted_cs = [1, 2, 3]
    #         else:
    #             restricted_cs = cs  # 如果没有限制，c 可以为 [1, 2, 3, 4]
    #
    #         for c in restricted_cs:
    #             run_model(l, dim, c)
    run_model(1, 128, 3)

