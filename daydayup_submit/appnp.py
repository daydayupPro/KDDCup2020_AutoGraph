"""Training an APPNP model."""

import random
import torch
import numpy as np
from tqdm import trange
from appnp_layer import APPNPModel
import scipy.sparse as sp
import itertools
import time

class APPNPTrainer(object):
    """
    Method to train PPNP/APPNP model.
    """
    def __init__(self, graph, features, target, train_mask, test_mask, sp_density, learning_rate=0.01, lambd=5e-3, epochs=600, early_stopping_rounds=500, model_name="exact", iterations=5, alpha=0.16, layers=[64,64], dropout=0.6):
        """
        :param args: Arguments object.
        :param graph: Networkx graph.
        :param features: Feature matrix.
        :param target: Target vector with labels.
        """
        self.device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.graph = graph
        self.features = features
        self.target = target
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.sp_density = sp_density
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.model_name = model_name
        self.iterations = iterations
        self.alpha = alpha
        self.layers = layers
        self.dropout = dropout
        self.create_model()
        self.train_test_split()
        self.transfer_node_sets()
        self.process_features()
        self.transfer_features()


    def create_model(self):
        """
        Defining a model and transfering it to GPU/CPU.
        """
        self.node_count = self.features.shape[0]
        self.number_of_labels = np.max(self.target)+1
        self.number_of_features = self.features.shape[1]

        self.model = APPNPModel(self.number_of_labels,
                                self.number_of_features,
                                self.graph,
                                self.device,
                                self.model_name,
                                self.iterations,
                                self.alpha,
                                self.layers,
                                self.dropout)

        self.model = self.model.to(self.device)

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        self.train_nodes = self.train_mask
        self.test_nodes = self.test_mask

    def transfer_node_sets(self):
        """
        Transfering the node sets to the device.
        """
        self.train_nodes = torch.LongTensor(self.train_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(self.test_nodes).to(self.device)

    def process_features(self):
        """
        Creating a sparse feature matrix and a vector for the target labels.
        """
        dot_features = sp.dok_matrix(self.features)
        indice_f = list(map(list,zip(*list(dot_features.keys()))))
        values = list(dot_features.values())

        self.feature_indices = torch.LongTensor(indice_f)
        self.feature_values = torch.FloatTensor(values)
        self.target = torch.LongTensor(self.target)

    def transfer_features(self):
        """
        Transfering the features and the target matrix to the device.
        """
        self.target = self.target.to(self.device)
        self.feature_indices = self.feature_indices.to(self.device)
        self.feature_values = self.feature_values.to(self.device)

    def pred(self, index_set):
        self.model.eval()
        _, pred = self.model(self.feature_indices, self.feature_values).max(dim=1)
        return pred[index_set]

    def train_neural_network(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for _ in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            prediction = self.model(self.feature_indices, self.feature_values)
            loss = torch.nn.functional.nll_loss(prediction[self.train_nodes],
                                                self.target[self.train_nodes])
            loss = loss+(self.lambd/2)*(torch.sum(self.model.layer_2.weight_matrix**2))
            if _%100 == 0:
                print(_, loss)
            loss.backward()
            self.optimizer.step()
        
        y_pred = self.pred(self.test_nodes)

        return y_pred.cpu().numpy().flatten()
