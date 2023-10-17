import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from utils import random_mini_batches

from data import data_prepration



X_train, Y1_train, Y2_train, Y3_train, X_valid, Y1_valid, Y2_valid, Y3_valid, X_test, Y1_test, Y2_test, Y3_test = data_prepration()
num_inputs = X_train.shape[1]
input_size, feature_size = X_train.shape

epoch3 = 1000
mb_size = 250
cost1tr_3 = []
cost2tr_3 = []
cost3tr_3 = []
cost1D_3 = []
cost2D_3 = []
cost3D_3 = []
cost1ts_3 = []
cost2ts_3 = []
cost3ts_3 = []
costtr_3 = []
costD_3 = []
costts_3 = []

class NeuralNetwork(nn.Module):
    def __init__(self,hidden_nuron1, hidden_nuron2, hidden_nuron3, num_inputs=feature_size ):
        super().__init__()
        self.fc = nn.Linear(num_inputs, hidden_nuron1)
        self.fc1 = nn.Linear(hidden_nuron1, hidden_nuron2)
        self.fc2 = nn.Linear(hidden_nuron2, hidden_nuron3)
        self.fc3 = nn.Linear(hidden_nuron3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x1 = self.fc3(x)
        x2 = self.fc3(x)
        x3 = self.fc3(x)
        return x1, x2, x3
    
MTL3 = NeuralNetwork(hidden_nuron1=104, hidden_nuron2=55, hidden_nuron3=23, num_inputs=feature_size)

#LR3 = 0.01
#gamma3 = 1
#step_size3 = 1000
optimizer3 = torch.optim.Adam(MTL3.parameters(), lr=0.005793433148440374, weight_decay= 2e-2)
#scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=step_size3, gamma=gamma3)
loss_func3 = nn.MSELoss()

def main():
    # Train model
    print()
    print("Training...")
    
    for it in range(epoch3):
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        epoch_cost3 = 0
        num_minibatches = int(input_size / mb_size)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train,Y3_train, mb_size)):
            X_batch, Y1_batch, Y2_batch, Y3_batch  = minibatch
            predict_x1, predict_x2, predict_x3= MTL3(X_batch)
            loss1 = loss_func3(predict_x1, Y1_batch.view(-1,1))
            loss2 = loss_func3(predict_x2, Y2_batch.view(-1,1))
            loss3 = loss_func3(predict_x3, Y3_batch.view(-1,1))
            loss = loss1 + loss2 + loss3
            optimizer3.zero_grad()
            loss.backward()
            optimizer3.step()
            #scheduler3.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
            epoch_cost3 = epoch_cost3 + (loss3 / num_minibatches)
        costtr_3.append(torch.mean(epoch_cost))
        cost1tr_3.append(torch.mean(epoch_cost1))
        cost2tr_3.append(torch.mean(epoch_cost2))
        cost3tr_3.append(torch.mean(epoch_cost3))
        #MTL3.eval()
        with torch.no_grad():
            predict_x1, predict_x2, predict_x3 = MTL3(X_valid)
            l1D = loss_func3(predict_x1, Y1_valid.view(-1,1))
            l2D = loss_func3(predict_x2, Y2_valid.view(-1,1))
            l3D = loss_func3(predict_x3, Y3_valid.view(-1,1))
            cost1D_3.append(l1D)
            cost2D_3.append(l2D)
            cost3D_3.append(l3D)
            costD_3.append(l1D+l2D+l3D)
        MTL3.eval()  
        with torch.no_grad():
            predict = MTL3(X_test)    
            l1ts = loss_func3(predict[0], Y1_test.view(-1,1))
            l2ts = loss_func3(predict[1], Y2_test.view(-1,1))
            l3ts = loss_func3(predict[2], Y3_test.view(-1,1))
            cost1ts_3.append(l1ts)
            cost2ts_3.append(l2ts)
            cost3ts_3.append(l3ts)
            costts_3.append(l1ts+l2ts)
        print("Epoch: ", it, " Training Loss: ", cost1tr_3[-1], cost2tr_3[-1], cost3tr_3[-1], " Validation Loss: ", cost1D_3[-1], cost2D_3[-1], cost2D_3[-1], " Test Loss: ", cost1ts_3[-1], cost2ts_3[-1], cost3ts_3[-1])
                    
if __name__ == '__main__':
    main()    