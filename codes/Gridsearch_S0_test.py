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
import torch.utils.data as Data
from model.grid_cross5_model_S0_86 import model_S0

import pickle
import dgl
import csv

from torch_geometric.data import Batch

from cross5_S0_data_helper import construct_graph, get_kg
from sklearn.preprocessing import StandardScaler
import warnings
import warnings
import warnings

warnings.filterwarnings("ignore")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True  # 保证每次训练的结果一致
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def do_compute_metrics(probas_pred, target):
    y_pred_train1 = []
    y_label_train = np.array(target)
    y_label_train=y_label_train.reshape((-1))
    y_pred_train = np.array(probas_pred).reshape((-1, 86))
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


def test(model, test_data_loader, index,device):
    src, dst, rel, hr2eid, rt2eid = construct_graph(index)
    graph = get_kg(src, dst, rel, device)

    test_probas_pred=[]
    test_ground_truth=[]
    val_drug1s = []
    val_drug2s = []
    with torch.no_grad():
        for batch in test_data_loader:
            drug1s, drug2s, labels = batch
            outs, allemb = model(drug1s, drug2s, graph, drug1s, drug2s)

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
        #np.save('alldrugemb_cross5.npy', allemb.detach().cpu().numpy())
        test_acc, test_f1, test_recall, test_precision, test_kappa,predlabls,truelabels = do_compute_metrics(test_probas_pred, test_ground_truth)
        print("test:", test_acc, test_f1, test_recall, test_precision, test_kappa)
        predlabls += 1
        truelabels += 1
        concatenated = np.concatenate(
            (val_drug1s[:, np.newaxis], val_drug2s[:, np.newaxis], truelabels[:, np.newaxis], predlabls[:, np.newaxis]),
            axis=1)
        # 保存到CSV文件
        #np.savetxt('MGFF_5.1.10_results.csv', concatenated, delimiter=',')
        with open('MGFF_testS0_cross5_label_results.csv', 'a') as f:
            np.savetxt(f, concatenated, delimiter=',')
######################### Parameters ######################
parser = argparse.ArgumentParser(description="Parser for DDI")
parser.add_argument('--zhongzi', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=256)


if __name__ == '__main__':
    args = parser.parse_args()

    def run_model():

        batch_size = args.batchsize#params['n_batch']
        for index in range(1,6):
            df_ddi_test = pd.read_csv('../data/cross5datasets/S0/S0_idx_test_fold_' + str(index) + '.csv',header=None)
            #df_ddi_test = pd.read_csv('../data/cross5datasets/Galantamine.csv',header=None)
            #df_ddi_test = pd.read_csv('../data/cross5datasets/S0/AllCaseStudyDatasets.csv',header=0)
            test_drug1 = torch.tensor(np.array(df_ddi_test.values[:, 0], dtype=int)).to(device=device)
            test_drug2 = torch.tensor(np.array(df_ddi_test.values[:, 1], dtype=int)).to(device=device)
            test_label = torch.tensor(np.array(df_ddi_test.values[:, 2], dtype=int)).to(device=device)

            testdata = Data.TensorDataset(test_drug1, test_drug2, test_label)

            print(f"validating with {len(testdata)}")

            test_data_loader = Data.DataLoader(testdata, batch_size=batch_size, shuffle=True, drop_last=False)

            model = model_S0(device,index,128,1,3).to(device=device)

            model.load_state_dict(torch.load('./S0_MGFF_cross12813_'+str(index)+'.pkl'), strict=False)
            model.eval()
            print("test!", index)
            test(model, test_data_loader, index,device)

    run_model()






