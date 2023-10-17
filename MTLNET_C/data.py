# Synthetic Data Generator for Machine Learning models
import numpy as np
import random
import math
import torch
from utils import train_test_split_for_dataframe, train_test_split_for_array, numpy_to_pytorch, data_preprocessing

seed = 42
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def data_preparation(num_feature, rho, num_row):
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
    
    # Apply data pre processing on synthetics by using minimaxsclar in return you will get numpyarray
    inputs_df, target_df1, target_df2 = data_preprocessing(input_data=X, target_label1=Y1, target_label2=Y2)
    
    # Split the data into train, and test set by using train_test_split_for_array
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split_for_array(inputs_df, target_df1, target_df2, test_size=0.2, random_state=42)

   # Convert the numpy array to torch tensor
    X_train, y1_train, y2_train = numpy_to_pytorch(X_train, y1_train, y2_train)
    X_test, y1_test, y2_test = numpy_to_pytorch(X_test, y1_test, y2_test)
    
    return X_train, y1_train, y2_train, X_test, y1_test, y2_test