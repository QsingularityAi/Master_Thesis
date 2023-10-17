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
epoch1 = 1000
mb_size1 = 50
hidden_nuron1=151
hidden_nuron2=47
hidden_nuron3=9
Total_costtr = []
Task1_cost1tr = []
Task2_cost2tr = []

Valtotal_costD = []
ValTask1_cost1D = []
ValTask2_cost2D = []

testtotal_costD = []
testTask1_cost1D = []
testTask2_cost2D = []

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
        return x1, x2
MTL3 = NeuralNetwork(hidden_nuron1, hidden_nuron2, hidden_nuron3, num_inputs=feature_size)    
gamma3 = .99
step_size3 = 100
optimizer3 = torch.optim.Adam(MTL3.parameters(), lr=0.0002778240111230875, weight_decay= 6.83426960998647e-10 )
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=step_size3, gamma=gamma3)
# loss_func3 = nn.L1Loss()
loss_func3 = nn.MSELoss()    
    
def main():
    
    
    print()
    print("Training...")
    # Training
    for it in range(epoch1):
        Total_costtr = []
        Task1_cost1tr = []
        Task2_cost2tr = []
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        num_minibatches = int(input_size / mb_size1)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, mb_size1)):
        # forward
            X_batch, Y1_batch, Y2_batch  = minibatch
            predict = MTL3(X_batch)
        # compute loss
            loss1 = loss_func3(predict[0], Y1_batch.view(-1,1))
            loss2 = loss_func3(predict[1], Y2_batch.view(-1,1))
            loss = loss1 + loss2
        # backward
            optimizer3.zero_grad()
            loss.backward()
            optimizer3.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
        Total_costtr.append(torch.mean(epoch_cost))
        Task1_cost1tr.append(torch.mean(epoch_cost1))
        Task2_cost2tr.append(torch.mean(epoch_cost2))
        # Evaluation
        with torch.no_grad():
            predict = MTL3(X_valid)
            l1D = loss_func3(predict[0], Y1_valid.view(-1,1))
            l2D = loss_func3(predict[1], Y2_valid.view(-1,1))
            ValTask1_cost1D.append(l1D)
            ValTask2_cost2D.append(l2D)
            Valtotal_costD.append(l1D+l2D)
        MTL3.eval()
        with torch.no_grad():
            predict = MTL3(X_test)
            l1ts = loss_func3(predict[0], Y1_test.view(-1,1))
            l2ts = loss_func3(predict[1], Y2_test.view(-1,1))
            testTask1_cost1D.append(l1ts)
            testTask2_cost2D.append(l2ts)
            testtotal_costD.append(l1ts+l2ts)
            
        print("Epoch: ", it, " Training Loss: ", Task1_cost1tr[-1], Task2_cost2tr[-1]," Val Loss: ", ValTask1_cost1D[-1], ValTask2_cost2D[-1], " test Loss: ", testTask1_cost1D[-1], testTask2_cost2D[-1])     
            
if __name__ == "__main__":
    main()
        