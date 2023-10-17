import numpy as np
import torch
import random
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

seed = 42
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def random_mini_batches(X_batch, Y1_batch, Y2_batch, mini_batch_size = 5, seed = 42): 
    # Creating the mini-batches
    np.random.seed(seed)            
    m = X_batch.shape[0]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X_batch[permutation,:] 
    shuffled_Y1 = Y1_batch[permutation]
    shuffled_Y2 = Y2_batch[permutation]
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y1 = shuffled_Y1[k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y2 = shuffled_Y2[k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y1, mini_batch_Y2)
        mini_batches.append(mini_batch)
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[Lower : Lower + Upper, :]
        mini_batch_Y1 = shuffled_Y1[Lower : Lower + Upper]
        mini_batch_Y2 = shuffled_Y2[Lower : Lower + Upper]
        mini_batch = (mini_batch_X, mini_batch_Y1, mini_batch_Y2)
        mini_batches.append(mini_batch)
    
    return mini_batches


### for split pandas dataframe

def train_test_split_for_dataframe(X,y1,y2,test_size=0.2,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_train_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_train_samples)
    test_set_size = int(n_train_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y1.iloc[train_indices], y1.iloc[test_indices],y2.iloc[train_indices], y2.iloc[test_indices]

### for split numpy array

def train_test_split_for_array(X,y1,y2,test_size=0.2,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_train_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_train_samples)
    test_set_size = int(n_train_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y1[train_indices], y1[test_indices], y2[train_indices], y2[test_indices]

### convert numpy to pytorch tensor

def numpy_to_pytorch(input, target1, target2):
    x = torch.from_numpy(input).float()
    y1 = torch.tensor(target1).float()
    y2 = torch.tensor(target2).float()
    return x, y1, y2

### Applying data pre-processing on synthetic data

def data_preprocessing(inputs_df, target_df1, target_df2):
    

    numeric_cols = inputs_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols1 = target_df1.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols2 = target_df2.select_dtypes(include=['int64', 'float64']).columns.tolist()

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler3 = MinMaxScaler()

    scaler1.fit(inputs_df[numeric_cols])
    scaler2.fit(target_df1[numeric_cols1])
    scaler3.fit(target_df2[numeric_cols2])

    inputs_df[numeric_cols] = scaler1.transform(inputs_df[numeric_cols])
    target_df1[numeric_cols1] = scaler2.transform(target_df1[numeric_cols1])
    target_df2[numeric_cols2] = scaler3.transform(target_df2[numeric_cols2])
    
    X = inputs_df.to_numpy()
    Y1 = target_df1.to_numpy()
    Y2 = target_df2.to_numpy()
    
    return X, Y1, Y2
