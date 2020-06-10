"""the simple baseline for autograph"""

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader
import networkx as nx

import random
from collections import Counter

from utils import normalize_features
import scipy.sparse as sp
from appnp import APPNPTrainer
from daydayup_model import GCNTrainer, TAGTrainer, XGBTrainer

from scipy import stats

from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
from daydayup_private_features import dayday_feature, dayday_feature_old


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(1234)


class Model:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def generate_pyg_data_appnp(self, data, x, edge_index):


        graph = nx.from_edgelist(edge_index)
        features= normalize_features(x)

        num_nodes = features.shape[0]

        target = np.zeros(num_nodes, dtype=np.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        target[inds] = train_y

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        return graph, features, target, train_indices, test_indices

    def generate_pyg_data_gcn(self, data, x, edge_index):
    
        x = torch.tensor(x, dtype=torch.float)

        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

        edge_weight = data['edge_file']['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask

        return data
    

    def train_predict(self, data, time_budget, n_class, schema):        

        flag_feature = 1
        sp_density = 0.0
        flag_zero = 1
        
        x = data['fea_table']
        
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x), dtype=np.float)
            flag_feature = 0
        else:
            x.replace([np.inf, -np.inf], np.nan, inplace=True)
            x.fillna(0, inplace=True)
            x = x.drop('node_index', axis=1).to_numpy()
            x_max = x.max()
            x_min = x.min()
            if x_max == x_min:
                x = np.arange(x.shape[0])
                x = np.array(pd.get_dummies(x), dtype=np.float)
                flag_zero = 0
            else:
                sp_density = np.count_nonzero(x)/x.size*1.
                x = x.astype(np.float)

        label_counter = Counter(data['train_label']['label'])
        label_most_common_1 = label_counter.most_common(1)[0][0]
        label_len = len(label_counter)

        df = data['edge_file']
        edge_count = df.shape[0]
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        
        if sp_density >= 0.1:
            data = self.generate_pyg_data_gcn(data, x, edge_index)
            lr_lst = [0.005, 0.005, 0.005, 0.005, 0.005]
            my_epochs = 700
            pred = []
            if all([edge_count >= 4e5, edge_count <= 7e5]):
                my_epochs = 500
            elif all([edge_count > 7e5, edge_count < 15e5]):
                lr_lst = [0.005, 0.005, 0.005]
                my_epochs = 400
            elif edge_count >= 15e5:
                lr_lst = [0.005]
                my_epochs = 500
            for lr in lr_lst:
                trainer = GCNTrainer(data, lr=lr, weight_decay=2e-4, epochs=my_epochs)
                temp = trainer.train_nn()
                pred.append(temp)
            pred = stats.mode(pred)[0][0]
        
        elif all([flag_feature == 0, label_len<=3]):
            data = self.generate_pyg_data_gcn(data, x, edge_index)
            try:
                my_epochs = 500
                if edge_count >= 14e5:
                    my_epochs = 400
                trainer = TAGTrainer(data, lr=0.018, weight_decay=2e-4, epochs=my_epochs, hidden=16, dropout=0.5)
                pred = trainer.train_nn()
            except:
                lr_lst = [0.005, 0.005, 0.005, 0.005, 0.005]
                my_epochs = 700
                pred = []
                if all([edge_count >= 4e5, edge_count <= 7e5]):
                    my_epochs = 500
                elif all([edge_count > 7e5, edge_count < 15e5]):
                    lr_lst = [0.005, 0.005, 0.005]
                    my_epochs = 400
                elif edge_count >= 15e5:
                    lr_lst = [0.005]
                    my_epochs = 500
                for lr in lr_lst:
                    trainer = GCNTrainer(data, lr=lr, weight_decay=2e-4, epochs=my_epochs)
                    temp = trainer.train_nn()
                    pred.append(temp)
                pred = stats.mode(pred)[0][0]

        elif all([flag_feature == 0, label_len>3]):
            print("you are best")
            train_indices=data['train_indices']
            test_indices=data['test_indices']

            feature_neighbor = dayday_feature_old(data)
            train_y = data['train_label']['label'].to_numpy()
            train_x = feature_neighbor[train_indices]
            test_x = feature_neighbor[test_indices]

            trainer = XGBTrainer(train_x, train_y, test_x,  n_class, max_depth=6, subsample=0.7, colsample_bytree=0.7, random_state=0, n_jobs=3)
            pred = trainer.train_nn()

        elif flag_zero == 0:           
            train_indices=data['train_indices']
            test_indices=data['test_indices']

            feature_neighbor = dayday_feature(data, n_class=n_class, label_most_common_1=label_most_common_1)
            train_y = data['train_label']['label'].to_numpy()
            train_x = feature_neighbor[train_indices]
            test_x = feature_neighbor[test_indices]

            trainer = XGBTrainer(train_x, train_y, test_x,  n_class, n_jobs=3)
            pred = trainer.train_nn()

        else:          
            graph, features, target, train_mask, test_mask = self.generate_pyg_data_appnp(data, x, edge_index)
            trainer = APPNPTrainer(graph, features, target, train_mask, test_mask, sp_density, learning_rate=0.012, lambd=2.5e-3, epochs=600, model_name="exact", iterations=5, alpha=0.31, layers=[64,64], dropout=0.6)
            pred = trainer.train_neural_network()

        return pred

