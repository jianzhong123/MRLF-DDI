#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
#from decoder import ConvE
import numpy as np
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import csv
#from model.ConvE import *

class SE_GNN(nn.Module):
    def __init__(self,h_dim=128,l=1):
        super().__init__()
        self.n_ent = 569
        self.n_rel = 65
        self.dim=h_dim

        #self.fc=nn.Sequential(nn.Linear(1280,1280),nn.ReLU(),nn.Linear(1280,256),nn.ReLU())
        # gnn layer
        self.kg_n_layer = l
        # relation SE layer
        self.edge_layers = nn.ModuleList([EdgeLayer(self.dim) for _ in range(self.kg_n_layer)])
        # entity SE layer
        self.node_layers = nn.ModuleList([NodeLayer(self.dim) for _ in range(self.kg_n_layer)])
        # triple SE layer
        self.comp_layers = nn.ModuleList([CompLayer(self.dim) for _ in range(self.kg_n_layer)])
        rel_w =  Parameter(torch.zeros(self.dim*l, self.dim))
        self.rel_w = xavier_normal_(rel_w)
        # relation embedding for aggregation
        param = Parameter(torch.zeros(self.n_rel * 2, self.dim))
        self.relparam = xavier_normal_(param)
        self.rel_embs = nn.ParameterList([self.relparam for _ in range(self.kg_n_layer)])

        # relation embedding for prediction
        param = Parameter(torch.zeros(h_dim * self.kg_n_layer, self.dim))
        self.rel_w = xavier_normal_(param)

        self.ent_drop = nn.Dropout(0.1)
        self.rel_drop = nn.Dropout(0.1)
        self.fc =nn.Linear(h_dim,self.dim)

    def forward(self,drug1_id, drug2_id,  kg,ent_emb):

        # aggregate embedding
        agg_emb = self.aggragate_emb(kg,ent_emb)
        drug1_emb = agg_emb[drug1_id.long()]
        drug2_emb = agg_emb[drug2_id.long()]
        return drug1_emb,drug2_emb#,gnnloss


    def aggragate_emb(self, kg,ent_emb):

        rel_emb_list = []
        count=0
        for edge_layer, node_layer, comp_layer, rel_emb in zip(self.edge_layers, self.node_layers, self.comp_layers, self.rel_embs):
            count = count+1
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            if count==1 and self.kg_n_layer!=1:
               ent_emb=self.fc(ent_emb)#[h_id]
            rel_emb=rel_emb#[r_id]
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, rel_emb)

            x_min = ent_emb.min(dim=0).values
            x_max = ent_emb.max(dim=0).values
            edge_ent_emb = (edge_ent_emb - x_min) / (x_max - x_min)
            node_ent_emb = (node_ent_emb - x_min) / (x_max - x_min)
            comp_ent_emb = (comp_ent_emb - x_min) / (x_max - x_min)
            agg_emb = ent_emb +edge_ent_emb + node_ent_emb + comp_ent_emb#
            # 进行最小-最大归一化

            #agg_emb = torch.cat((edge_ent_emb,node_ent_emb,comp_ent_emb),dim=1)
            #rel_emb_list.append(rel_emb)

        #pred_rel_emb = torch.cat(rel_emb_list, dim=1)
        #pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        return agg_emb#,pred_rel_emb


class CompLayer(nn.Module):
    def __init__(self, h_dim=128):
        super().__init__()
        self.n_ent = 569
        self.n_rel = 65
        self.comp_op = 'mul'#self.comp_op in ['add', 'mul']

        param = Parameter(torch.zeros(h_dim, h_dim))
        self.neigh_w = xavier_normal_(param)
        self.act = nn.Tanh()

        self.bn = torch.nn.BatchNorm1d(h_dim)


    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():

            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            # neihgbor entity and relation composition
            if self.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class NodeLayer(nn.Module):
    def __init__(self, h_dim=128):
        super().__init__()

        self.n_ent = 569
        self.n_rel = 65
        param = Parameter(torch.zeros(h_dim, h_dim))
        self.neigh_w = xavier_normal_(param)

        self.act = nn.Tanh()

        self.bn = torch.nn.BatchNorm1d(h_dim)


    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # attention
            #使用点乘方式计算每条边上节点嵌入 'emb' 的点积，并将结果存储在边数据 'norm' 中。
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb

    def loss(self, score, label):
        loss = self.bce(score, label)
        return loss

#实现了一个基于知识图谱的边层处理，包括边的注意力计算、信息聚合和邻居节点的嵌入表示变换
class EdgeLayer(nn.Module):
    def __init__(self, h_dim=128):
        super().__init__()
        self.n_ent = 569
        self.n_rel = 65

        param = Parameter(torch.zeros(h_dim, h_dim))
        self.neigh_w = xavier_normal_(param)
        self.act = nn.Tanh()

        self.bn = torch.nn.BatchNorm1d(h_dim)

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb#
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # attention
            #使用 fn.e_dot_v 函数计算每条边上的注意力权重，结果存储在 kg.edata['norm'] 中
            #fn.e_dot_v 是一个函数，用于计算边的嵌入 'emb' 与关联节点的嵌入 'emb' 之间的点积
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            #对计算得到的注意力权重进行 softmax 归一化
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            #将每条边的嵌入乘以对应的注意力权重
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            #使用 update_all 方法更新所有节点的邻居信息，其中 fn.copy_e 用于复制边的嵌入到消息 'm'，
            #然后 fn.sum 将所有接收到的消息求和，存储在节点数据 'neigh' 中。
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
               neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
