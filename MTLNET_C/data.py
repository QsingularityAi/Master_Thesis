# Synthetic Data Generator for Machine Learning models
import numpy as np
import random
import math
import torch

seed = 42
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def data_preparation(num_feature, rho):
    num_row = 12000
    c = 0.3
    u1 = np.random.randn(num_feature)
    u1 = (u1 - np.mean(u1)) / (np.std(u1) * np.sqrt(num_feature))
    u2 = np.random.randn(num_feature)
    u2 -= u2.dot(u1) * u1
    u2 /= np.linalg.norm(u2)

    # k = np.random.randn(num_feature)
    # u1 = np.random.randn(num_feature)
    # u1 -= u1.dot(k) * k / np.linalg.norm(k)**2
    # u1 /= np.linalg.norm(u1)
    # k /= np.linalg.norm(k)
    # u2 = k
    w1 = c * u1
    w2 = c * (rho * u1 + np.sqrt((1 - rho**2))*u2)
    X = np.random.normal(0, 1, (num_row, num_feature))
    eps1 = np.random.normal(0, 0.01)
    eps2 = np.random.normal(0, 0.01)
    Y1 = np.matmul(X, w1) + np.sin(np.matmul(X, w1))+eps1
    Y2 = np.matmul(X, w2) + np.sin(np.matmul(X, w2))+eps2
    split = list(np.random.permutation(num_row))

    X_train = X[split[0:10000],:]
    Y1_train = Y1[split[0:10000]]
    Y2_train = Y2[split[0:10000]]
    X_valid = X[10000:11000,:]
    Y1_valid = Y1[10000:11000]
    Y2_valid = Y2[10000:11000]
    X_test = X[11000:12000,:]
    Y1_test = Y1[11000:12000]
    Y2_test = Y2[11000:12000]

    # Convert the numpy array to torch tensor
    
    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    X_test = torch.from_numpy(X_test).float()

    Y1_train = torch.tensor(Y1_train).float()
    Y2_train = torch.tensor(Y2_train).float()

    Y1_valid = torch.tensor(Y1_valid).float()
    Y2_valid = torch.tensor(Y2_valid).float()

    Y1_test = torch.tensor(Y1_test).float()
    Y2_test = torch.tensor(Y2_test).float()
    
    return X_train, Y1_train, Y2_train, X_valid, Y1_valid, Y2_valid, X_test, Y1_test, Y2_test