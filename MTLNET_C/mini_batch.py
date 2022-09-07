#### Creating minibatch generator for the training set.
import numpy as np
import random
import math

def random_mini_batches(X_batch, Y1_batch, Y2_batch, mini_batch_size = 10, seed = 42): 
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