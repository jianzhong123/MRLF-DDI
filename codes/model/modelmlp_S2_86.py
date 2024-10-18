import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import  numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from model.crossModel import GCNModel
from model.SEmodel import SE_GNN
from sklearn.decomposition import PCA

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(2929),#
            nn.Linear(2929, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512, 2929)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class model_S0(torch.nn.Module):
    def __init__(self,device):#in_dim,hidden_dim,out_dim,
        super(model_S0, self).__init__()
        self.inter = GCNModel()#.to(device=device)
        self.gnn = SE_GNN(256)#.to(device)

        self.mlp = nn.ModuleList([nn.Linear(768, 768),
                                  nn.ELU(),
                                  nn.Linear(768, 86)
                                  ])

        self.bn1 = torch.nn.BatchNorm1d(768)#

        self.autocoder=Autoencoder()
        self.criterion = nn.MSELoss()

        pub_path = '../data/drugfeatures/drugs1559_pubchem86.npy'
        np_pub = np.load(pub_path)
        pub = torch.tensor(np_pub).to(device).float()

        morgen_path = '../data/drugfeatures/drugs1559_morgen86.npy'
        np_morgen = np.load(morgen_path)
        morgen = torch.tensor(np_morgen).to(device).float()

        alignn_path = '../data/drugfeatures/S12_drugs1559_alignn86.npy'  #
        np_alignn = np.load(alignn_path)
        alignn = torch.tensor(np_alignn).to(device).float()
        x_min = alignn.min(dim=0).values
        x_max = alignn.max(dim=0).values
        # 进行最小-最大归一化
        alignn = (alignn - x_min) / (x_max - x_min)

        mol_path = '../data/drugfeatures/drugs1559_mol86.npy'
        np_mol = np.load(mol_path)
        mol = torch.tensor(np_mol).to(device).float()

        input_features = torch.cat((pub, morgen, alignn, mol), dim=1)
        pca = PCA(n_components=256)
        features = pca.fit_transform(input_features.cpu().detach().numpy())
        self.pcafeatures = torch.Tensor(features).float().to(device)
        self.features1 = nn.Embedding(1559, 2929)
        self.fc = nn.Linear(256, 128)
    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward(self, drug1s,drug2s, graph,simdrug1s,simdrug2s):
        input_features=self.features1.weight
        target = input_features

        en_emb1, decoded = self.autocoder(input_features)
        autoloss = self.criterion(decoded, target)

        en_emb1=en_emb1+self.pcafeatures
        #en_emb1.requires_grad_(True)
        drug1_emb = en_emb1[drug1s.long()]
        drug2_emb = en_emb1[drug2s.long()]

        drug1_global_emb, drug2_global_emb = self.gnn(simdrug1s, simdrug2s, graph, en_emb1)

        drug1_global_emb += drug1_emb
        drug2_global_emb += drug2_emb

        #fusion2=torch.cat((drug1_global_emb,drug2_global_emb),dim=1)#w/o CL
        #fusion2 = self.inter.generate_fusion_feature(drug1_emb, drug2_emb)
        fusion2 = self.inter.generate_fusion_feature(drug1_global_emb, drug2_global_emb)
        fusion2 = self.bn1(fusion2)
        drug1_emb = self.fc(drug1_emb)
        drug2_emb = self.fc(drug2_emb)

        #all = torch.cat((fusion2, drug1_emb, drug2_emb), 1)

        all=fusion2
        #print("all", all.shape)
        out = self.MLP(all, 3)
        return out,autoloss,en_emb1
