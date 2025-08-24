from jws.dataprocessing.medmnist3d import load_train_test
import numpy as np

dset = 'vessel'

X_train, y_train, X_test, y_test, X_val, y_val = load_train_test(dset, False)
print(X_train.shape)

# X_train[X_train > 0] = 1
print(np.unique(np.max(X_train, axis=(1,2,3))))