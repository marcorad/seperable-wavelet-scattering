import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import pickle as pkl
from sklearn import metrics
import torch
from jws.dataprocessing.medmnist3d import DATASETS
from sklearn.preprocessing import MultiLabelBinarizer

import torch

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
from tqdm import tqdm

# torch.set_default_device('cuda')



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
    def __init__(self, model : nn.Module) -> None:
        self.model = model
        self.model.cuda()
        
    def train(self, X_train, y_train, X_val, y_val, n_epochs=100, lr=1e-3): 
        n_classes = len(torch.unique(y_train))     
        self.n_classes = n_classes   
        self.le = LabelEncoder()
        y_train = torch.from_numpy(self.le.fit_transform(y_train))
        
        y_train_weights = torch.zeros(len(self.le.classes_))
        
        for v in y_train.unique():
            y_train_weights[v] = torch.sum(y_train == v)
            
        y_train_weights = 1 / y_train_weights.type(torch.float32)
        y_train_weights /= y_train_weights.sum()      
        
        
        y_val = torch.from_numpy(self.le.transform(y_val))
        y_train = nn.functional.one_hot(y_train).type(torch.float32) if n_classes > 2 else y_train.type(torch.float32)
        
        
        X_val = X_val.cuda()
        y_val = nn.functional.one_hot(y_val).type(torch.float32).cuda() if n_classes > 2 else y_val.type(torch.float32).cuda()
        generator = torch.Generator(device='cuda')
        self.batch_size = 256
        self.loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True, generator=generator)
        optim = torch.optim.Adam(params=self.model.parameters(), lr = lr)
        loss_fn = nn.CrossEntropyLoss(weight=y_train_weights) if n_classes > 2 else nn.BCELoss()
        
        best_val_loss = 1e12
        best_weights = None
        epoch_pbar = tqdm(range(n_epochs))
        for n in epoch_pbar:            
            self.model.train()
            with torch.set_grad_enabled(True):
                for x_batch, y_batch in self.loader:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                    y_pred = self.model(x_batch)
                    if n_classes == 2: 
                        y_pred = y_pred[:, 0]
                        loss = loss_fn(y_pred, y_batch.to(torch.float32))
                    else:
                        loss = loss_fn(y_pred, y_batch.to(torch.float32))
                    
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    
            #evaluate the validation set
            self.model.eval()
            with torch.set_grad_enabled(False):
                y_pred = self.model(X_val)
                if n_classes == 2: y_pred = y_pred[:, 0]
                test_loss = loss_fn(y_pred, y_val)  
                if n_classes == 2: test_loss = test_loss.sum()
                
                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                    best_weights = self.model.state_dict()
                epoch_pbar.set_description(f'Validation loss: {loss:.4f}')
                    
            
        # load the best model
        self.model.load_state_dict(best_weights)        
               
                
    def test_acc(self, X_test, y_test):
        self.model.eval()
        with torch.set_grad_enabled(False):
            y_test = torch.from_numpy(self.le.transform(y_test))
            y_test = nn.functional.one_hot(y_test.cuda()).type(torch.float32) if self.n_classes > 2 else y_test.cuda().type(torch.float32)
            X_test = X_test.cuda()
            y_pred = self.model(X_test)
            
            auc = metrics.roc_auc_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
            
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
        
        
 
if __name__ == '__main__':               


    EN_LDA_DR = False

    results = {}

    Q = 1
    Qs = [
        [[Q], [Q], [Q]],
        [[Q, Q], [Q, Q], [Q, Q]],
    ]
    ds = [
        # [2]*3,
        [4]*3,
        [6]*3,
        [8]*3
    ]

    for Q in Qs:
        for d in ds:
            for dset in DATASETS:
                fname = f'medmnist3d-cache/ws-{dset}-mnist3d-{Q=}-{d=}.pkl' #run medmnist3d_features.py before running this
                with open(fname, 'rb') as file:
                    X_train, y_train, X_test, y_test, X_val, y_val = pkl.load(file)
                    y_train = torch.from_numpy(y_train.astype(np.float32))
                    y_test = torch.from_numpy(y_test.astype(np.float32))
                    y_val = torch.from_numpy(y_val.astype(np.float32))
                    
                X_train = torch.reshape(X_train, (X_train.shape[0], -1))    
                X_test = torch.reshape(X_test, (X_test.shape[0], -1))  
                X_val = torch.reshape(X_val, (X_val.shape[0], -1)) 
                
                n_classes = len(torch.unique(y_train))             

                mu = 0 #torch.mean(X_train, axis=0)
                std = 1 #torch.std(X_train, axis=0)

                X_train = (X_train - mu)/std
                X_test = (X_test - mu)/std
                X_val = (X_val - mu)/std    
                
                net = DeepClassifier(X_train.shape[1],[512, 512, 256], n_classes)
                trainer = LinearTrainer(net)
                
                trainer.train(X_train, y_train, X_val, y_val, n_epochs=100, lr=1e-3)
                acc, auc = trainer.test_acc(X_test, y_test)
                
                if str(Q) not in results.keys(): results[str(Q)] = {}
                if str(d) not in results[str(Q)].keys(): results[str(Q)][str(d)] = {}
                results[str(Q)][str(d)][dset] = {
                    'acc': acc.item(),
                    'auc': auc
                }
                print(f'{dset} ({d=}, {Q=}): ACC={acc*100:.3f} AUC={auc:.3f}')
        
    
    
    with open('medmnist3d-results/results.json', 'w') as file:
        import json
        json.dump(results, file, indent=4)