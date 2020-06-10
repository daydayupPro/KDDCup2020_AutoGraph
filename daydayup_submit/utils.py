"""Utilities for data manipulation."""

import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
import math

def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(np.round(features, 5))

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def feature_reader(path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    features = json.load(open(path))
    features = {int(k): [int(val) for val in v] for k, v in features.items()}
    return features

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"])
    return target

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=(node_count, node_count), dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph, alpha, model_name):
    """
    Creating  apropagation matrix.
    :param graph: NetworkX graph.
    :param alpha: Teleport parameter.
    :param model: Type of model exact or approximate.
    :return propagator: Propagator matrix Dense torch matrix /
    dict with indices and values for sparse multiplication.
    """
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    if model_name == "exact":
        propagator = (I-(1-alpha)*A_tilde_hat).todense()
        propagator = alpha*torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        indices = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1).T
        propagator["indices"] = torch.LongTensor(indices)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
