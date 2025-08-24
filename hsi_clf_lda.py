import numpy as np

from jws.scattering.joint_scattering import JointScattering
import torch
from jws.scattering.config import cfg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from jws.dataprocessing import hsi
import gc

Q = 1
d_hyp_configs = [4, 6, 8, 6, 8] #  
d_im_configs =  [2, 2, 2, 4, 4] #  

cfg.cuda()

# cfg.set_alpha(Q,    2.5, False)
# # cfg.set_alpha(Q,    2.0, True)
# cfg.set_beta(Q,     2.5)
# cfg.FORCE_ANALYTICITY = True
# cfg.NORMALISE_LITTLE_WOOD_PALEY = False

hsi_data = hsi.load()

for hsi_im in hsi_data[:]:
    gc.collect()
    print(hsi_im['name'])
    if hsi_im['name'] not in ['indian_pines_corrected', 'KSC', 'paviaU', 'Botswana']: continue
    
    print('\n------------------------')
    print(hsi_im['name'])
    print('------------------------\n')

    res = []

    for d_im, d_hyp in zip(d_im_configs, d_hyp_configs):
        gc.collect()
        
        
        X = hsi_im['image'].astype(np.float32)
        labels = hsi_im['labels']
        
        im_shape = X.shape

        print(X.shape, labels.shape)
        
        torch.cuda.empty_cache()

        # if d_im == 2:   cfg.set_alpha(Q,    1.8, True)
        # else:           cfg.set_alpha(Q,    2.3, True)
        sws = JointScattering(list(X.shape), [d_im, d_im, d_hyp], [[Q], [Q], [Q]], allow_ds=[False, False, True], remove_highly_corr_filter=True)
        X = torch.from_numpy(X[None, :, :, :]).cuda()
        s = sws.scattering(X).cpu().numpy()[0, :, :, :, :]        
        
        # unpadding disabled when downsampling is disabled for any dimension, so do it manually
        nb = d_im // 2
        s = s[:, nb:(nb+X.shape[1]), nb:(nb+X.shape[2]), 1:-1]
        print(s.shape)


        x = np.swapaxes(s, 0, 2).swapaxes(0, 1).reshape((im_shape[0], im_shape[1], -1)).reshape((im_shape[0]*im_shape[1], -1))
        del s
        labels = labels.reshape(im_shape[0]*im_shape[1])
        
        # drop unlabeled pixels
        idx = labels != 0
        X = x[idx, :]
        y = labels[idx]
        
        def split(y: np.ndarray, n=15):
            N = len(y)
            train_idx = [False for _ in range(N)]
            idx = [i for i in range(N)]
            for yu in np.unique(y):
                train_samples = np.nonzero(y == yu)[0]            
                np.random.shuffle(train_samples)
                train_samples = train_samples[:n] 
                for k in train_samples:
                    train_idx[k] = True
                    
            train_idx_ret = []
            test_idx_ret = []
            for i in range(N):
                if train_idx[i]:
                    train_idx_ret.append(i)
                else:
                    test_idx_ret.append(i)        
            return train_idx_ret, test_idx_ret        
        
        gc.collect()
        # continue
        sizes = [15, -1]
        for sz in sizes:
            acc = []
            Ntrails = 10
            gc.collect()
            n = 15
            if sz == -1:
                n = 0.1 if hsi_im['name'] == 'indian_pines_corrected' else 0.05
            print(n)
            for i in tqdm(list(np.random.randint(0,high=100000, size=(Ntrails,)))):
                # print(i)
                if n < 1:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=i, shuffle=True, stratify=y)
                else:
                    train_idx, test_idx = split(y)
                    X_train, y_train = X[train_idx, :], y[train_idx]
                    X_test, y_test = X[test_idx, :], y[test_idx]
                
                
                mu =  0 #np.mean(X_train, axis=0)
                std =  1 #np.std(X_train, axis=0)
                X_train = (X_train - mu) / std
                X_test = (X_test - mu) / std
                c, counts = np.unique(y, return_counts=True)
                c = len(c)

                clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc.append(np.sum(y_pred == y_test) / len(y_pred))
                
            oa = np.array(acc)*100
            print('Randomised trials overall accuracy (%) result')
            print(f'd = {(d_im, d_im, d_hyp)}, Q = {(Q, Q, Q)}: {np.mean(oa):.2f} +- {np.std(oa):.2f} min {np.min(oa):.2f} max {np.max(oa):.2f}')
            
            res += [{
                'Q': (Q, Q, Q),
                'd': (d_im, d_im, d_hyp),
                'mean': np.mean(oa),
                'std': np.std(oa),
                'min': np.min(oa),
                'max': np.max(oa),
                'raw': oa.tolist(),
                'nfeat': X_train.shape[1],
                'ntrain': n
            }]
        
    import json
    with open(f"hsi-results/{hsi_im['name']}-results.json", 'w') as file:
        json.dump(res, file, indent=4)
    

