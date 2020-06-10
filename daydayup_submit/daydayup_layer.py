import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, APPNP, ChebConv, ARMAConv, GATConv, SGConv, TAGConv, SAGEConv, DNAConv
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from utils import reset, uniform

import random
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)
EPS = 1e-15

class GCN(torch.nn.Module):

    def __init__(self, features_num, num_class, num_layers, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class My_APPNP(torch.nn.Module):
    
    def __init__(self, K=5, alpha=0.16, hidden=128, num_features=16, num_class=2, aggr='add', num_layers=2):
        super(My_APPNP, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.prop1 = APPNP(K, alpha, aggr)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class Cheb_Net(torch.nn.Module):
    def __init__(self, features_num, num_class, hidden, num_hops, dropout):
        super(Cheb_Net, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(features_num, hidden, num_hops)
        self.conv2 = ChebConv(hidden, num_class, num_hops)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class ARMA_Net(torch.nn.Module):
    def __init__(self, features_num, num_class, hidden, num_stacks, num_layers, shared_weights, dropout, skip_dropout):
        super(ARMA_Net, self).__init__()
        self.dropout = dropout
        self.conv1 = ARMAConv(
            features_num,
            hidden,
            num_stacks,
            num_layers,
            shared_weights,
            dropout=skip_dropout)
        self.conv2 = ARMAConv(
            hidden,
            num_class,
            num_stacks,
            num_layers,
            shared_weights,
            dropout=skip_dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GAT_Net(torch.nn.Module):
    def __init__(self, features_num, num_class, hidden, heads, output_heads, concat, dropout):
        super(GAT_Net, self).__init__()
        self.dropout = dropout
        self.first_lin = Linear(features_num, hidden)
        self.conv1 = GATConv(
            in_channels=hidden,
            out_channels=hidden,
            concat=concat,
            heads=heads,
            dropout=dropout)
        self.conv2 = GATConv(
            in_channels=hidden * heads,
            out_channels=num_class,
            concat=concat,
            heads=output_heads,
            dropout=dropout)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SGC_Net(torch.nn.Module):
    def __init__(self, features_num, num_class, K, cached):
        super(SGC_Net, self).__init__()
        self.conv1 = SGConv(features_num, num_class, K, cached)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class TAG_Net(torch.nn.Module):
    def __init__(self, features_num, num_class, hidden, dropout):
        super(TAG_Net, self).__init__()
        self.dropout = dropout
        self.conv1 = TAGConv(features_num, hidden)
        self.conv2 = TAGConv(hidden, num_class)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class DNA_Net(torch.nn.Module):
    def __init__(self, features_num, num_class, num_layers, hidden, heads, groups ,dropout):
        super(DNA_Net, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.lin1 = Linear(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(DNAConv(hidden, heads, groups, dropout=0.8, cached=True))
        self.lin2 = Linear(hidden, num_class)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_all = x.view(-1, 1, self.hidden)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
