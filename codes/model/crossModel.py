# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from time import time
import pandas
import pandas as pd

import torch.optim as optim

from collections import OrderedDict

import argparse
import logging

import os
import random

import torch.utils.data as Data

import copy

import torch as th

EMB_INIT_EPS = 2.0
gamma = 12.0

class GCNModel(nn.Module):
    def __init__(self, ):
        super(GCNModel, self).__init__()
        self.entity_dim = 128
        self.fusion_type = 'init_double'
        self.mess_dropout = 0.1


        # fusion type
        if self.fusion_type == 'init_double':

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
                 nn.MaxPool2d((2, 2)), )#nn.ReLU() nn.BatchNorm2d(8),
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
                nn.MaxPool2d((2, 2)),
                #nn.ReLU()
            )# nn.BatchNorm2d(8),

            self.fc1 = nn.Linear(256,128)
            self.fc2 = nn.Linear(256, 256)
            #self.fc1 = nn.Linear(64,64)
            #self.fc2 = nn.Linear(64,64)
            self.fc2_global = nn.Sequential(
                nn.Linear(23112, self.entity_dim),
                nn.ReLU(True)#128:23112,64:5448,256:95304
                )#
            self.fc2_global_reverse = nn.Sequential(
                nn.Linear(23112, self.entity_dim),#23112
                nn.ReLU(True)
            )

    def generate_fusion_feature(self, drug1_emb, drug2_emb):
        # we focus on approved drug
        global embedding_data
        global embedding_data_reverse
        #print(drug2_emb.shape)
        drug11_emb=self.fc1(drug1_emb)
        drug22_emb=self.fc1(drug2_emb)

        drug1_emb = self.fc2(drug1_emb)
        drug2_emb = self.fc2(drug2_emb)
        structure_embed_reshape = drug11_emb.unsqueeze(-1)  # batch_size * embed_dim * 1
        entity_embed_reshape = drug22_emb.unsqueeze(-1)  # batch_size * embed_dim * 1

        entity_matrix = structure_embed_reshape * entity_embed_reshape.permute(
            (0, 2, 1))  # batch_size * embed_dim * embed_dim

        entity_matrix_reverse = entity_embed_reshape * structure_embed_reshape.permute(
            (0, 2, 1))  # batch_size * embed_dim * embed_dim

        entity_global = entity_matrix.view(entity_matrix.size(0), -1)#[2322, 10000]

        entity_global_reverse = entity_matrix_reverse.view(entity_matrix.size(0), -1)

        entity_matrix_reshape = entity_matrix.unsqueeze(1)
        entity_data=entity_matrix_reshape


        out = self.conv1(entity_data)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        #out = self.fc2(out)#[128,100]

        #embedding_data = torch.cat((embedding_data, out), 0)

        global_local_before = torch.cat((out, entity_global), 1)

        #print("global",global_local_before.shape)
        cross_embedding_pre = self.fc2_global(global_local_before)
        #cross_embedding_pre = self.bn(cross_embedding_pre)
        #print("cross",cross_embedding_pre.shape)
        # another reverse part
        entity_matrix_reshape_reverse = entity_matrix_reverse.unsqueeze(1)
        entity_reverse=entity_matrix_reshape_reverse

        out = self.conv1(entity_reverse)
        out = self.conv2(out)

        out = out.view(out.size(0), -1)

        global_local_before_reverse = torch.cat((out, entity_global_reverse), 1)
        cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)

        out_concat = torch.cat((drug1_emb,cross_embedding_pre, cross_embedding_pre_reverse,drug2_emb), 1)

        return out_concat

