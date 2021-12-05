import numpy as np
from HGSVM import HyperSVM
from HGSVM import rbf
import scipy.io as sio
if __name__ == "__main__":
    print('########################################Loading Data###########################################')
    data_path = './Twomoons.mat'
    idx_path = './Twomoons_0.1lab.mat'
    data_mat = sio.loadmat(data_path)
    idx_mat = sio.loadmat(idx_path)
    X=data_mat["X"]
    Y = np.ndarray.flatten(np.int64(np.transpose(data_mat["Y"])))
    idxLabs=np.ndarray.flatten(np.int64(np.transpose(idx_mat["idxLabs"])))
    idxUnls=np.ndarray.flatten(np.int64(np.transpose(idx_mat["idxUnls"])))
    idxTrain=np.ndarray.flatten(np.int64(np.transpose(idx_mat["idxTrain"])))
    idxTest=np.ndarray.flatten(np.int64(np.transpose(idx_mat["idxTest"])))
    opt = {'neighbor_mode': 'connectivity',
           'n_neighbor': 5,
           't': 1,
           'kernel_function': rbf,
           'kernel_parameters': {'gamma': 10},
           'gamma_A': 0.03125,
           'gamma_I': 1}
    accuracy = []
    k = 0
    # training data, testing data
    X_Train=X[idxTrain]
    Y_Train=Y[idxTrain]
    X_Test=X[idxTest]
    Y_Test=Y[idxTest]
    #labeled data，unlabeled data
    Xl=X_Train[idxLabs]
    Yl=Y_Train[idxLabs]
    Xu=X_Train[idxUnls]
    s = HyperSVM(opt)
    s.fit(Xl, Yl, Xu)
    Y_ = s.decision_function(X_Test)
    Y_pre = np.ones(X_Test.shape[0])
    Y_pre[Y_ < 0] = -1
    accuracy.append(np.nonzero(Y_pre == Y_Test)[0].shape[0] / X_Test.shape[0] * 100.)

    print('---------------------------------------------------------------------')
    print('Accuracy：')
    print(accuracy)

