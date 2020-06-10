from daydayup_layer import GCN, My_APPNP, Cheb_Net, ARMA_Net, GAT_Net, SGC_Net, TAG_Net, DNA_Net
import torch
import torch.nn.functional as F
import lightgbm as lgb
from torch_geometric.nn import GCNConv, Node2Vec 
from torch.nn import PReLU
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import time
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter

class GCNTrainer(object):
    def __init__(self, data, lr=0.005, weight_decay=2e-4, epochs=700, features_num=16, num_class=2, num_layers=3, hidden=128):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.num_layers = num_layers
        self.hidden = hidden
        self.features_num = features_num
        self.num_class = num_class

    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = GCN(features_num=self.features_num, num_class=self.num_class, hidden=self.hidden, num_layers=self.num_layers)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()


class MyAPPNPTrainer(object):
    def __init__(self, data, lr=0.005, weight_decay=2.5e-4, epochs=500):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = My_APPNP(num_features=self.features_num, num_class=self.num_class)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()
        

class ChebTrainer(object):
    def __init__(self, data, lr=0.005, weight_decay=5e-4, epochs=600, features_num=16, num_class=2, hidden=64, num_hops=3, dropout=0.5):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.features_num = features_num
        self.num_class = num_class
        self.hidden = hidden
        self.num_hops = num_hops
        self.dropout = dropout


    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = Cheb_Net(features_num=self.features_num, num_class=self.num_class, hidden=self.hidden, num_hops=self.num_hops, dropout=self.dropout)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()


class ARMATrainer(object):
    def __init__(self, data, lr=0.027, weight_decay=5e-4, epochs=700, features_num=16, num_class=2, hidden=16, num_stacks=1, num_layers=1, shared_weights=True, dropout=0.5, skip_dropout=0.75):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.features_num = features_num
        self.num_class = num_class
        self.hidden = hidden
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        self.dropout = dropout
        self.skip_dropout = skip_dropout


    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = ARMA_Net(features_num=self.features_num, num_class=self.num_class, hidden=self.hidden, num_stacks=self.num_stacks, num_layers=self.num_layers, shared_weights=self.shared_weights, dropout=self.dropout, skip_dropout=self.skip_dropout)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()


class GATTrainer(object):
    def __init__(self, data, lr=0.005, weight_decay=2e-4, epochs=600, features_num=16, num_class=2, hidden=16, heads=3, output_heads=1, concat=True, dropout=0.6):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.features_num = features_num
        self.num_class = num_class
        self.hidden = hidden
        self.heads = heads
        self.output_heads = output_heads
        self.concat = concat
        self.dropout = dropout

    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = GAT_Net(features_num=self.features_num, num_class=self.num_class, hidden=self.hidden, heads=self.heads, output_heads=self.output_heads, concat=self.concat, dropout=self.dropout)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()


class SGCTrainer(object):
    def __init__(self, data, lr=0.005, weight_decay=5e-4, epochs=800, features_num=16, num_class=2, K=3, cached=True):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.features_num = features_num
        self.num_class = num_class
        self.K = K
        self.cached = cached


    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = SGC_Net(features_num=self.features_num, num_class=self.num_class, K=self.K, cached=self.cached)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()

class TAGTrainer(object):
    def __init__(self, data, lr=0.018, weight_decay=2e-4, epochs=500, features_num=16, num_class=2, hidden=16, dropout=0.5):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.features_num = features_num
        self.num_class = num_class
        self.hidden = hidden
        self.dropout = dropout

    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = TAG_Net(features_num=self.features_num, num_class=self.num_class, hidden=self.hidden, dropout=self.dropout)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()

class DNATrainer(object):
    def __init__(self, data, lr=0.005, weight_decay=5e-4, epochs=500, features_num=16, num_class=2, num_layers=5, hidden=128, heads=4, groups=16 ,dropout=0.5):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.features_num = features_num
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden = hidden
        self.heads = heads
        self.groups = groups
        self.dropout = dropout

    def train_nn(self):
        self.features_num = self.data.x.size()[1]
        self.num_class = int(max(self.data.y)) + 1
        self.model = DNA_Net(features_num=self.features_num, num_class=self.num_class, num_layers=self.num_layers, hidden=self.hidden, heads=self.heads, groups=self.groups ,dropout=self.dropout)
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            if epoch % 100 == 0:
                print(epoch, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data)[self.data.test_mask].max(1)[1]
        
        return pred.cpu().numpy().flatten()


class N2VTrainer(object):
    def __init__(self, data, lr=0.001, epochs=4):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.lr = lr
        self.epochs = epochs

    def test(self, train_z, train_y, test_z, solver='lbfgs', multi_class='auto', *args, **kwargs):
        clf = LogisticRegression(
            solver=solver, 
            multi_class=multi_class,
            *args,
            **kwargs
        )
        clf.fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        pred = clf.predict(test_z.detach().cpu().numpy())
        return pred

    def train_nn(self):
        self.model = Node2Vec(
            self.data.num_nodes,
            embedding_dim=128, 
            walk_length=20,
            context_size=10, 
            walks_per_node=10
        )
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loader = DataLoader(torch.arange(self.data.num_nodes), batch_size=128, shuffle=True)

        for epoch in range(1, self.epochs+1):
            t1 = time.time()        
            self.model.train()
            total_loss = 0
            for subset in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(self.data.edge_index, subset.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            total_loss = total_loss/len(self.loader)
            print("epoch: %d, time elapsed: %.2f, loss: %.5f" % (epoch, time.time()-t1, total_loss))

        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.arange(self.data.num_nodes, device=self.device))
        
        return z


class GBDTTrainer(object):
    def __init__(self, train_x, train_y, test_x,  n_class):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.n_class = n_class

    def train_nn(self):

        clf = GradientBoostingClassifier(
            loss="deviance", # "exponential" 
            learning_rate=0.1, 
            n_estimators=100, 
            criterion="friedman_mse",
            min_samples_split=2, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0.,
            max_depth=5, 
            min_impurity_decrease=0., 
            min_impurity_split=1e-7,
            subsample=1.0, 
            max_features=None, # "auto" "sqrt" "log2"
            random_state=1234, 
            verbose=0, 
            max_leaf_nodes=None,
            warm_start=False, 
            presort='auto',
            validation_fraction=0.1, 
            n_iter_no_change=None,
            tol=1e-4
        )
        clf.fit(self.train_x, self.train_y)
        pred = clf.predict(self.test_x)

        return pred

class XGBTrainer(object):
    def __init__(self, 
                train_x, 
                train_y, 
                test_x,  
                n_class,
                prob=False,
                max_depth=5, 
                learning_rate=0.08,
                n_estimators=100, 
                silent=True,
                objective="multi:softmax", 
                booster='gbtree',
                n_jobs=1, 
                nthread=None, 
                gamma=0, 
                min_child_weight=1,
                max_delta_step=0, 
                subsample=0.6, 
                colsample_bytree=0.6, 
                colsample_bylevel=1,
                reg_alpha=1.0, 
                reg_lambda=0.8, 
                scale_pos_weight=1,
                base_score=0.5, 
                random_state=214, 
                seed=None, 
                missing=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.n_class = n_class
        self.prob = prob
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.booster = booster
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step 
        self.subsample = subsample 
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda 
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.seed = seed
        self.missing = missing

    def train_nn(self):
        model = XGBClassifier(
            max_depth=self.max_depth, 
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators, 
            silent=self.silent,
            objective=self.objective, 
            booster=self.booster,
            n_jobs=self.n_jobs, 
            nthread=self.nthread, 
            gamma=self.gamma, 
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step, 
            subsample=self.subsample, 
            colsample_bytree=self.colsample_bytree, 
            colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha, 
            reg_lambda=self.reg_lambda, 
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score, 
            random_state=self.random_state, 
            seed=self.seed, 
            missing=self.missing)

        model.fit(self.train_x, self.train_y)
        if self.prob:
            pred = model.predict_proba(self.test_x)
        else:
            pred = model.predict(self.test_x)
        
        return pred
