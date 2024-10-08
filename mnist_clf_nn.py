
import sys


sys.path.append('../python')

import sepws.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT


from sepws.scattering.separable_scattering import SeparableScattering
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

from sepws.scattering.config import cfg

cfg.cuda()
cfg.set_beta(1, 2.5)
cfg.set_alpha(1, 2.5)

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import pickle as pkl
from sklearn import metrics
import torch
from sepws.dataprocessing.medmnist3d import DATASETS

import torch

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder

import torch.nn as nn

class DeepClassifier(nn.Module):
    def __init__(self, input_shape, hidden_sizes, num_classes) -> None:
        super().__init__()
        self.lin_in = nn.Linear(input_shape, hidden_sizes[0])
        self.bn_in = nn.BatchNorm1d(hidden_sizes[0])
        self.bn = []
        self.lin_hid = []
        for i in range(len(hidden_sizes)-1):
            self.lin_hid.append(nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i+1]))
            self.bn.append(nn.BatchNorm1d(hidden_sizes[i+1]))
        self.lin_out = nn.Linear(hidden_sizes[-1], num_classes) if num_classes > 2 else nn.Linear(hidden_sizes[-1], 1)
        self.soft_max = nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid()
        self.non_linearity = nn.ReLU()
        
    def forward(self, x):
        y = self.non_linearity(self.bn_in(self.lin_in(x)))
        for l, b in zip(self.lin_hid, self.bn):
            y = self.non_linearity(b(l(y)))         
        y = self.soft_max(self.lin_out(y))
        return y     
        
        
    
class LinearTrainer:
    def __init__(self, model) -> None:
        self.model = model
        
    def train(self, X_train, y_train, X_val, y_val, n_epochs=100, lr=1e-3): 
        n_classes = len(torch.unique(y_train))     
        self.n_classes = n_classes   
        self.le = LabelEncoder()
        y_train = torch.from_numpy(self.le.fit_transform(y_train))
        
        y_train_weights = torch.zeros(len(self.le.classes_))
        
        for v in y_train.unique():
            y_train_weights[v] = torch.sum(y_train == v)
            
        print('Train counts: ', y_train_weights.tolist())
        y_train_weights = 1 / y_train_weights.type(torch.float32)
        y_train_weights /= y_train_weights.sum()      
        
        
        y_val = torch.from_numpy(self.le.transform(y_val))
        y_train = nn.functional.one_hot(y_train).type(torch.float32) if n_classes > 2 else y_train.type(torch.float32)
        
        
        X_val = X_val.cuda()
        y_val = nn.functional.one_hot(y_val).type(torch.float32).cuda() if n_classes > 2 else y_val.type(torch.float32).cuda()
        generator = torch.Generator(device='cuda')
        self.batch_size = 256
        self.loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, generator=generator, shuffle=True)
        optim = torch.optim.Adam(params=self.model.parameters(), lr = lr)
        loss_fn = nn.CrossEntropyLoss(weight=y_train_weights) if n_classes > 2 else nn.BCELoss()
        
        prev_test_loss = 1e12
        loss_avg = None
        for n in range(n_epochs):            
            self.model.train()
            with torch.set_grad_enabled(True):
                for x_batch, y_batch in self.loader:
                    x_batch = x_batch.cuda()
                    # x_batch *= torch.randn_like(x_batch)*0.05 + 1
                    y_batch = y_batch.cuda()
                    y_pred = self.model(x_batch)
                    if n_classes == 2: 
                        y_pred = y_pred[:, 0]
                        # weight = torch.zeros_like(y_batch)
                        # weight[y_batch == 0] = y_train_weights[0]
                        # weight[y_batch == 1] = y_train_weights[1]
                        # loss_fn = nn.BCELoss(weight=weight)
                        loss = loss_fn(y_pred, y_batch.to(torch.float32))
                    else:
                        loss = loss_fn(y_pred, y_batch.to(torch.float32))
                    
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    
            #print accuracy
            self.model.eval()
            # if n_classes == 2: 
            #     weight = torch.zeros_like(y_val)
            #     weight[y_val == 0] = y_train_weights[0]
            #     weight[y_val == 1] = y_train_weights[1]
            #     loss_fn = nn.BCELoss(weight=weight)
            with torch.set_grad_enabled(False):
                y_pred = self.model(X_val)
                if n_classes == 2: y_pred = y_pred[:, 0]
                test_loss = loss_fn(y_pred, y_val)  
                if n_classes == 2: test_loss = test_loss.sum()
                
                if loss_avg == None: 
                    loss_avg = test_loss
                    prev_loss_avg = test_loss
                else:
                    p = 0.9
                    prev_loss_avg = loss_avg
                    loss_avg = loss_avg * (1-p) + p*test_loss             
                if prev_loss_avg < loss_avg: return   
                
                if n_classes > 2:     
                    y_true = torch.argmax(y_val, dim=1)
                    y_pred = torch.argmax(y_pred, dim=1)
                else:                    
                    y_pred = y_pred > 0.5
                    y_true = y_val  > 0.5  
                
                print(f'Epoch {n} validation accuracy: {torch.sum(y_true == y_pred) / y_true.shape[0] * 100: .2f} (loss={test_loss: .4f}) ')
                
    def test_acc(self, X_test, y_test):
        self.model.eval()
        with torch.set_grad_enabled(False):
            y_test = torch.from_numpy(self.le.transform(y_test))
            y_test = nn.functional.one_hot(y_test.cuda()).type(torch.float32) if self.n_classes > 2 else y_test.cuda().type(torch.float32)
            X_test = X_test.cuda()
            y_pred = self.model(X_test)
            
            auc = metrics.roc_auc_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), multi_class='ovo')
            
            y_true = torch.argmax(y_test, dim=1) if self.n_classes > 2 else y_test.cuda()
            y_pred = torch.argmax(y_pred, dim=1) if self.n_classes > 2 else y_pred[:,0].cuda()
            if self.n_classes == 2:
                thresh = 0.5
                y_true = y_true > thresh
                y_pred = y_pred > thresh
        return torch.sum(y_true == y_pred) / y_true.shape[0], auc
    
    def num_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


from sklearn.preprocessing import normalize

torch.cuda.empty_cache()
from kymatio.torch import Scattering2D

d = [8]*2
print(d)

cfg.set_alpha(1,    2.5, False)
cfg.set_alpha(1,    1.8, True)
cfg.set_beta(1,     2.5)

Q = [[1,1], [1,1]]

ws = SeparableScattering([28, 28], d, Q)


from sklearn import datasets, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from mnist import MNIST

N_train = 40000

mndata = MNIST('../python-mnist/data') #requires the python-mnist repo (https://pypi.org/project/python-mnist/) to be in the same directory as this repo
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
X_train = torch.from_numpy(np.array(X_train).reshape((-1, 28, 28))).type(cfg.REAL_DTYPE)[0:N_train, :, :]
y_train = np.array(y_train)[0:N_train]
X_test = torch.from_numpy(np.array(X_test).reshape((-1, 28, 28))).type(cfg.REAL_DTYPE)
y_test = np.array(y_test)

torch.cuda.empty_cache()

#extract features with SWS
norm = False
t0 = time()
S_train_sep = ws.scattering(X_train.to(cfg.DEVICE), normalise=norm).cpu()
S_test_sep  = ws.scattering(X_test.to(cfg.DEVICE), normalise=norm).cpu()
torch.cuda.synchronize()
t1 = time()


S_train_sep = torch.log(S_train_sep)
S_test_sep = torch.log(S_test_sep)
print("Sep Scattering took {:.2f} ms".format((t1 - t0)*1000))
print(S_train_sep.shape)
# exit()

torch.cuda.empty_cache()
ws_2d = Scattering2D(J=3, shape=(28, 28), max_order=len(Q[0]))
ws_2d.cuda()

t0 = time()
S_train_2d: torch.Tensor = ws_2d.scattering(X_train.cuda())
S_test_2d: torch.Tensor  = ws_2d.scattering(X_test.cuda())   
torch.cuda.synchronize()
t1 = time()
print("2D Scattering took {:.2f} ms".format((t1 - t0)*1000))
S_train_2d = S_train_2d.swapaxes(1, -1)
S_test_2d = S_test_2d.swapaxes(1, -1)
S_train_2d = torch.log(S_train_2d)
S_test_2d = torch.log(S_test_2d)

print(S_train_2d.shape)
print('2D DEVICE', S_test_2d.device)

Nval = 0

# exit()

# #flatten
S_train_sep = S_train_sep.reshape(S_train_sep.shape[0], np.prod(S_train_sep.shape[1:]))
S_val_sep = S_train_sep[0:Nval, ...]
S_train_sep = S_train_sep[Nval:, ...]
S_test_sep = S_test_sep.reshape(S_test_sep.shape[0], np.prod(S_test_sep.shape[1:]))

# mu = torch.mean(S_train_sep, dim=0)
# std = torch.std(S_train_sep, dim=0)
# S_train_sep = (S_train_sep - mu)/std
# S_val_sep = (S_val_sep - mu)/std
# S_test_sep = (S_test_sep - mu)/std

S_train_2d = S_train_2d.reshape(S_train_2d.shape[0], np.prod(S_train_2d.shape[1:]))
S_val_2d = S_train_2d[0:Nval, ...]
S_train_2d = S_train_2d[Nval:, ...]
S_test_2d = S_test_2d.reshape(S_test_2d.shape[0], np.prod(S_test_2d.shape[1:]))

y_val = torch.from_numpy(y_train[0:Nval])
y_train = torch.from_numpy(y_train[Nval:])
y_test = torch.from_numpy(y_test)


lda = LDA(solver='eigen', shrinkage='auto')
lda.fit(S_train_2d.cpu().numpy(), y_train.numpy())
y_pred = lda.predict(S_test_2d.cpu().numpy())
print('2D', metrics.accuracy_score(y_pred, y_test.numpy()))

lda = LDA(solver='eigen', shrinkage='auto')
lda.fit(S_train_sep.cpu().numpy(), y_train.numpy())
y_pred = lda.predict(S_test_sep.cpu().numpy())
print('Sep', metrics.accuracy_score(y_pred, y_test.numpy()))

exit()

print(S_train_sep.shape)
print(S_train_2d.shape)

sep_err = []
_2d_err = []

for i in range(1):

    net = DeepClassifier(S_train_sep.shape[1],[128, 64, 32], 10)
    trainer = LinearTrainer(net)
    print("SEP PARAMETERS", trainer.num_trainable_parameters())
    trainer.train(S_train_sep, y_train, S_val_sep, y_val, n_epochs=100, lr=1e-4)
    acc, _ = trainer.test_acc(S_test_sep, y_test)
    err_prc = (1-acc)*100
    sep_err.append(err_prc.item())
    print(f'SEP Test Err: {err_prc: .2f}')


    net = DeepClassifier(S_train_2d.shape[1],[128, 64, 32], 10)
    trainer = LinearTrainer(net)
    print("2D PARAMETERS", trainer.num_trainable_parameters())
    trainer.train(S_train_2d, y_train, S_val_2d, y_val, n_epochs=100, lr=1e-4)
    acc, _ = trainer.test_acc(S_test_2d, y_test)
    err_prc = (1-acc)*100   
    _2d_err.append(err_prc.item())
    print(f'2D Test Err: {err_prc: .2f}')
    
print(sep_err)
print(_2d_err)

print('SEP ERR: ', np.mean(sep_err), np.std(sep_err))
print('2D ERR: ', np.mean(_2d_err), np.std(_2d_err))

