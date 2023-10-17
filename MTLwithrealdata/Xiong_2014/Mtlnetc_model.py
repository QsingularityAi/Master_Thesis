import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from utils import random_mini_batches

from data import data_prepration


X_train, Y1_train, Y2_train, X_valid, Y1_valid, Y2_valid, X_test, Y1_test, Y2_test = data_prepration()
num_inputs = X_train.shape[1]
input_size, feature_size = X_train.shape
hidden_layer_size = 23
task_layesr1= 10
task_layesr2= 9
n_output_1 = 1
n_output_2 = 1
n_output_3 = 1
epochs = 1000
mb_size = 5
cost1tr_2 = []
cost2tr_2 = []
cost3tr_2 = []
cost4tr_2 = []
cost1D_2 = []
cost2D_2 = []
cost3D_2 = []
cost4D_2 = []
costtr_2 = []
costD_2 = []
cost11D = []
cost12D = []
cost11tr = []
cost12tr = []
cost1ts_2=[]
cost11ts=[]
cost12ts=[]
cost3ts_2=[]
cost4ts_2=[]
costts_2=[]

class MTL_NetworkC(nn.Module):
    def __init__(self):
        super(MTL_NetworkC, self).__init__()
        self.input_layer = nn.Sequential(
                           nn.Linear(feature_size, hidden_layer_size),
                           nn.LeakyReLU()
                           )
        self.hidden_layer_1 = nn.Sequential(
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.LeakyReLU()
                            )
        self.output_layer = nn.Sequential(nn.Linear(hidden_layer_size, n_output_1),
                            nn.Sigmoid())
        self.hidden_layer_2 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layesr1),
                            nn.LeakyReLU(),
                            nn.Linear(task_layesr1, task_layesr1),
                            nn.Sigmoid(),
                            nn.Linear(task_layesr1, n_output_2))
        self.hidden_layer_3 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layesr2),
                            nn.LeakyReLU(),
                            nn.Linear(task_layesr2, task_layesr2),
                            nn.Sigmoid(),
                            nn.Linear(task_layesr2, n_output_3))                    
    
    def forward(self, x):
        input_data = self.input_layer(x)
        out1 = self.output_layer(input_data)
        out2 = self.output_layer(input_data)
        out3 = self.hidden_layer_2(input_data)
        out4 = self.hidden_layer_3(input_data)
        
        return out1, out2, out3, out4


MTL2 = MTL_NetworkC()    
#gamma2 = .99
#step_size2 = 100
optimizer2 = torch.optim.Adam(MTL2.parameters(), lr=0.018081179255739082, weight_decay= 5.392541670648882e-06)
#scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size2, gamma=gamma2)
# loss_func2 = nn.L1Loss()
loss_func2 = nn.MSELoss()    
    
def main():
    # Train model
    print()
    print("Training...")
    for it in range(epochs):
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost11 = 0
        epoch_cost12 = 0
        epoch_cost3 = 0
        epoch_cost4 = 0
        num_minibatches = int(input_size / mb_size)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, mb_size)):
            X_batch, Y1_batch, Y2_batch  = minibatch
            Y1_pred, Y2_pred, Y3_pred, Y4_pred = MTL2(X_batch)
            loss1 = loss_func2(Y1_pred, Y1_batch.view(-1,1))
            loss2 = loss_func2(Y2_pred, Y2_batch.view(-1,1))
            loss3 = loss_func2(Y3_pred, Y1_batch.view(-1,1))
            loss4 = loss_func2(Y4_pred, Y2_batch.view(-1,1))
            loss5 = loss1+ loss2
            loss = loss5 + loss3 + loss4
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            #scheduler2.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss5 / num_minibatches)
            epoch_cost11 = epoch_cost11 + (loss1/ num_minibatches)
            epoch_cost12 = epoch_cost12 + (loss2 / num_minibatches)
            epoch_cost3 = epoch_cost3 + (loss3 / num_minibatches)
            epoch_cost4 = epoch_cost4 + (loss4 / num_minibatches)
        costtr_2.append(torch.mean(epoch_cost))
        cost1tr_2.append(torch.mean(epoch_cost1))
        cost11tr.append(torch.mean(epoch_cost11))
        cost12tr.append(torch.mean(epoch_cost12))
        cost3tr_2.append(torch.mean(epoch_cost3))
        cost4tr_2.append(torch.mean(epoch_cost4))
        with torch.no_grad():
            val1_predict, val2_predict, val3_predict, val4_predict = MTL2(X_valid)
            l1D = loss_func2(val1_predict, Y1_valid.view(-1,1))
            l2D = loss_func2(val2_predict, Y2_valid.view(-1,1))
            l3D = loss_func2(val3_predict, Y1_valid.view(-1,1))
            l4D = loss_func2(val4_predict, Y2_valid.view(-1,1))
            val_loss = l1D+l2D
            cost1D_2.append(val_loss)
            cost11D.append(l1D)
            cost12D.append(l2D)
            cost3D_2.append(l3D)
            cost4D_2.append(l4D)
            costD_2.append(val_loss+l3D+l4D)
        MTL2.eval()
        with torch.no_grad():
            test1_predict, test2_predict, test3_predict, test4_predict = MTL2(X_test)
            l1ts = loss_func2(test1_predict, Y1_test.view(-1,1))
            l2ts = loss_func2(test2_predict, Y2_test.view(-1,1))
            l3ts = loss_func2(test3_predict, Y1_test.view(-1,1))
            l4ts = loss_func2(test4_predict, Y2_test.view(-1,1))
            test_loss = l1ts+l2ts
            cost1ts_2.append(test_loss)
            cost11ts.append(l1ts)
            cost12ts.append(l2ts)
            cost3ts_2.append(l3ts)
            cost4ts_2.append(l4ts)
            costts_2.append(test_loss+l3ts+l4ts)    
           
        

        print("Epoch: ", it, " Training Loss: ", cost3tr_2[-1], cost4tr_2[-1], " Validation Loss: ", cost3D_2[-1], cost4D_2[-1], " test Loss: ", cost3ts_2[-1], cost4ts_2[-1])
                    
if __name__ == '__main__':
    main()    