import numpy as np
import torch
import torch.nn as nn
from data import data_preparation
from MTLpoly import polymerinformatic
from utils import random_mini_batches
import matplotlib.pyplot as plt

LR = 0.0002210714707748555
mb_size = 100
# Training loss
cost1tr = []
cost2tr = []
costtr = []
# Validation loss
cost1D = []
cost2D = []
costD = []

def main():
    print()
    print("Training...")
    X_train, Y1_train, Y2_train, X_valid, Y1_valid, Y2_valid = data_preparation(num_feature=100, rho=0.8, num_row = 10000)
    input_size, feature_size = X_train.shape
    MTL = polymerinformatic(feature_size=feature_size, hidden_layer1=3, hidden_layer2=120, hidden_layer3=27)
    optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
    # loss_func = nn.L1Loss()
    loss_func = nn.MSELoss()
    for it in range(500):
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        cost1tr = []
        cost2tr = []
        costtr = []
        num_minibatches = int(input_size / mb_size)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, mb_size)):
            X_batch, Y1_batch, Y2_batch  = minibatch
            predict = MTL(X_batch)
            loss1 = loss_func(predict[0], Y1_batch.view(-1,1))
            loss2 = loss_func(predict[1], Y2_batch.view(-1,1))
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
        costtr.append(torch.mean(epoch_cost))
        cost1tr.append(torch.mean(epoch_cost1))
        cost2tr.append(torch.mean(epoch_cost2))
        with torch.no_grad():
            predict = MTL(X_valid)
            l1D = loss_func(predict[0], Y1_valid.view(-1,1))
            l2D = loss_func(predict[1], Y2_valid.view(-1,1))
            cost1D = []
            cost2D = []
            costD = []
            cost1D.append(l1D)
            cost2D.append(l2D)
            costD.append(l1D+l2D)
        print("Epoch: ", it, " Train Loss1: ", cost1tr[-1], " Val Loss1: ", cost1D[-1], " Train Loss2: ", cost2tr[-1], " Val Loss2: ",  cost2D[-1]) 
        
        #print("Epoch: ", it, " Training Loss: ", costtr[-1], " Validation Loss: ", costD[-1])
       
                    
if __name__ == '__main__':
    main()

