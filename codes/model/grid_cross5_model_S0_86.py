import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import  numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from model.gridsearch_crossModel import GCNModel
from model.gridsearch_SEmodel import SE_GNN
from sklearn.decomposition import PCA

class model_S0(torch.nn.Module):
    def __init__(self,device,index,dim,l,c):#in_dim,hidden_dim,out_dim,
        super(model_S0, self).__init__()
        self.dim=dim
        self.inter = GCNModel(dim,c)#.to(device=device)
        self.gnn = SE_GNN(dim,l)#.to(device)

        self.mlp = nn.ModuleList([nn.Linear(dim*6,dim*6),#3072
                                  nn.ELU(),
                                  nn.Linear(dim*6, 86)
                                  ])

        self.bn1 = torch.nn.BatchNorm1d(dim*2)#
        self.bn2 = torch.nn.BatchNorm1d(dim*2)
        self.bn3 = torch.nn.BatchNorm1d(dim*2)
        pub_path = '../data/drugfeatures/drugs1559_pubchem86.npy'
        np_pub = np.load(pub_path)
        pub = torch.tensor(np_pub).to(device).float()

        morgen_path = '../data/drugfeatures/drugs1559_morgen86.npy'
        np_morgen = np.load(morgen_path)
        morgen = torch.tensor(np_morgen).to(device).float()

        alignn_path = '../data/drugfeatures/S0align_cross'+str(index)+'.npy'  #
        np_alignn = np.load(alignn_path)
        alignn = torch.tensor(np_alignn).to(device).float()
        x_min = alignn.min(dim=0).values
        x_max = alignn.max(dim=0).values
        # 进行最小-最大归一化
        alignn = (alignn - x_min) / (x_max - x_min)

        mol_path = '../data/drugfeatures/drugs1559_mol86.npy'
        np_mol = np.load(mol_path)
        mol = torch.tensor(np_mol).to(device).float()

        self.features = torch.cat((pub, morgen, alignn, mol), dim=1)

        self.features1 = nn.Embedding(1559, dim)#2929
        self.fc2 = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(2929, dim)
        self.gate = nn.Linear(2 * 128, 128)  # 输入拼接后的维度
        self.sigmoid = nn.Sigmoid()

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward(self, drug1s,drug2s, graph,simdrug1s,simdrug2s):
        input_features=self.features1.weight

        modalfeatures1 = self.fc1(self.features)

        en_emb1=input_features+modalfeatures1
        drug1_global_emb, drug2_global_emb = self.gnn(simdrug1s, simdrug2s, graph, en_emb1)

        drug1_emb = en_emb1[drug1s.long()]
        drug2_emb = en_emb1[drug2s.long()]
        combined1 = torch.cat((drug1_global_emb, drug1_emb), dim=1)
        combined2 = torch.cat((drug2_global_emb, drug2_emb), dim=1)
        gate_weight1 = self.sigmoid(self.gate(combined1))
        gate_weight2 = self.sigmoid(self.gate(combined2))
        drug1_global_emb =  drug1_emb + (1 - gate_weight1) * drug1_global_emb
        drug2_global_emb = drug2_emb + (1 - gate_weight2) * drug2_global_emb

        #drug1_global_emb += drug1_emb #
        #drug2_global_emb += drug2_emb#+=

        globalemb=torch.cat((drug1_global_emb,drug2_global_emb),dim=1)#w/o CL
        globalemb = self.bn1(globalemb)
        #interemb = self.inter.generate_fusion_feature(drug1_emb, drug2_emb)
        interemb = self.inter.generate_fusion_feature(drug1_global_emb, drug2_global_emb)
        interemb = self.bn2(interemb)
        #drugpais = torch.cat((drug1_emb,drug2_emb),dim=1)#w/o CL
        #drugpais = self.bn3(drugpais)
        all = torch.cat((globalemb, drug1_emb,drug2_emb,interemb), 1)
        #all=fusion2

        #print("all", all.shape)
        out = self.MLP(all, 3)
        return out,en_emb1
