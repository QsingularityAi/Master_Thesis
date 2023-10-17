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

hidden_layer_size = 85
task_layesr1= 15
task_layesr2= 11
task_layesr3= 5
n_output_1 = 1
n_output_2 = 1
n_output_3 = 1
n_output_4 = 1
epoch2 = 1000
mb_size = 50
cost_Total = []
cost_comb = []
cost_task1 = []
cost_task2 = []
cost_task3 = []
cost_Extask1 = []
cost_Extask2 = []
cost_Extask3 = []
cost_valcomb =[]
cost_val1 = []
cost_val2 = []
cost_val3 = []
cost_val4 = []
cost_val5 = []
cost_val6 = []
cost_Total_val = []
test_valcomb = []
test_val1= []
test_val2= []
test_val3= []
test_val4= []
test_val5= []
test_val6= []
test_Total_val= []



class MTL_NetworkC(nn.Module):
    def __init__(self, hidden_layer_size, task_layesr1, task_layesr2, task_layesr3, num_inputs=feature_size, n_output_1=1, n_output_2=1,n_output_3=1, n_output_4=1):
        super(MTL_NetworkC, self).__init__()
        self.task_layesr1 = task_layesr1
        self.task_layesr2 = task_layesr2
        self.task_layesr3 = task_layesr3
        self.hidden_layer_size = hidden_layer_size
        self.input_layer = nn.Sequential(
                           nn.Linear(num_inputs, hidden_layer_size),
                           nn.ReLU()
                           )
        self.hidden_layer_1 = nn.Sequential(
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.ReLU(),
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.ReLU()
                            )
        self.output_layer = nn.Sequential(nn.Linear(hidden_layer_size, n_output_1),
                            nn.ReLU())
        self.hidden_layer_2 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layesr1),
                            nn.ReLU(),
                            nn.Linear(task_layesr1, task_layesr1),
                            nn.ReLU(),
                            nn.Linear(task_layesr1, n_output_2))
        self.hidden_layer_3 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layesr2),
                            nn.ReLU(),
                            nn.Linear(task_layesr2, task_layesr2),
                            nn.ReLU(),
                            nn.Linear(task_layesr2, n_output_3))
        self.hidden_layer_4 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layesr3),
                            nn.Sigmoid(),
                            nn.Dropout(0.0),
                            nn.Linear(task_layesr3, task_layesr3),
                            nn.Sigmoid(),
                            nn.Linear(task_layesr3, n_output_4))                     
    
    def forward(self, x):
        input_data = self.input_layer(x)
        out1 = self.output_layer(input_data)
        out2 = self.output_layer(input_data)
        out3 = self.output_layer(input_data)
        out4 = self.hidden_layer_2(input_data)
        out5 = self.hidden_layer_3(input_data)
        out6 = self.hidden_layer_4(input_data)
        return out1, out2, out3, out4, out5, out6
    
MTL2 = MTL_NetworkC(task_layesr1, task_layesr2, task_layesr3, hidden_layer_size, num_inputs=feature_size, n_output_1=1, n_output_2=1,n_output_3=1, n_output_4=1)    
gamma2 = 1
step_size2 = 50
optimizer2 = torch.optim.Adam(MTL2.parameters(), lr=0.002014802185493212, weight_decay= 0.0003499303509955949)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size2, gamma=gamma2)
loss_func2 = nn.MSELoss()    


def main():
    # Train model
    print()
    print("Training...")
    for it in range(epoch2):
        Total_cost = 0
        Combine_cost= 0
        Task1_cost = 0
        Task2_cost = 0
        Task3_cost = 0
        Task1_With_Exlayer = 0
        Task2_With_Exlayer = 0
        Task3_With_Exlayer = 0
        num_minibatches = int(input_size / mb_size)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, Y3_train, mb_size)):
            X_batch, Y1_batch, Y2_batch, Y3_batch = minibatch
            Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred = MTL2(X_batch)
            loss1 = loss_func2(Y1_pred, Y1_batch.view(-1,1))
            loss2 = loss_func2(Y2_pred, Y2_batch.view(-1,1))
            loss3 = loss_func2(Y3_pred, Y3_batch.view(-1,1))
            loss4 = loss_func2(Y4_pred, Y1_batch.view(-1,1))
            loss5 = loss_func2(Y5_pred, Y2_batch.view(-1,1))
            loss6 = loss_func2(Y6_pred, Y3_batch.view(-1,1))
            loss7 = loss1+ loss2 + loss3
            loss = loss7 + loss4 + loss5 + loss6
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            scheduler2.step()
            Total_cost = Total_cost + (loss / num_minibatches)
            Combine_cost = Combine_cost + (loss7 / num_minibatches)
            Task1_cost = Task1_cost + (loss1/ num_minibatches)
            Task2_cost = Task2_cost + (loss2 / num_minibatches)
            Task3_cost = Task3_cost + (loss3 / num_minibatches)
            Task1_With_Exlayer = Task1_With_Exlayer + (loss4 / num_minibatches)
            Task2_With_Exlayer = Task2_With_Exlayer + (loss5 / num_minibatches)
            Task3_With_Exlayer = Task3_With_Exlayer + (loss6 / num_minibatches)
        cost_Total.append(torch.mean(Total_cost))
        cost_comb.append(torch.mean(Combine_cost))
        cost_task1.append(torch.mean(Task1_cost))
        cost_task2.append(torch.mean(Task2_cost))
        cost_task3.append(torch.mean(Task3_cost))
        cost_Extask1.append(torch.mean(Task1_With_Exlayer))
        cost_Extask2.append(torch.mean(Task2_With_Exlayer))
        cost_Extask3.append(torch.mean(Task3_With_Exlayer))
        with torch.no_grad():
            val1_predict, val2_predict, val3_predict, val4_predict, val5_predict, val6_predict = MTL2(X_valid)
            l1D = loss_func2(val1_predict, Y1_valid.view(-1,1))
            l2D = loss_func2(val2_predict, Y2_valid.view(-1,1))
            l3D = loss_func2(val3_predict, Y3_valid.view(-1,1))
            l4D = loss_func2(val4_predict, Y1_valid.view(-1,1))
            l5D = loss_func2(val5_predict, Y2_valid.view(-1,1))
            l6D = loss_func2(val6_predict, Y3_valid.view(-1,1))
            val_loss = l1D+l2D+l3D
            cost_valcomb.append(val_loss)
            cost_val1.append(l1D)
            cost_val2.append(l2D)
            cost_val3.append(l3D)
            cost_val4.append(l4D)
            cost_val5.append(l5D)
            cost_val6.append(l6D)
            cost_Total_val.append(val_loss+l4D+l5D+l6D)
        MTL2.eval()    
        with torch.no_grad():
            test1_predict, test2_predict, test3_predict, test4_predict, test5_predict, test6_predict = MTL2(X_test)
            l1ts = loss_func2(test1_predict, Y1_test.view(-1,1))
            l2ts = loss_func2(test2_predict, Y2_test.view(-1,1))
            l3ts = loss_func2(test3_predict, Y3_test.view(-1,1))
            l4ts = loss_func2(test4_predict, Y1_test.view(-1,1))
            l5ts = loss_func2(test5_predict, Y2_test.view(-1,1))
            l6ts = loss_func2(test6_predict, Y3_test.view(-1,1))
            test_loss = l1ts+l2ts+l3ts
            test_valcomb.append(test_loss)
            test_val1.append(l1ts)
            test_val2.append(l2ts)
            test_val3.append(l3ts)
            test_val4.append(l4ts)
            test_val5.append(l5ts)
            test_val6.append(l6ts)
            test_Total_val.append(test_loss+l4ts+l5ts+l6ts)    
        # print('Epoch: [{}/{}], Loss: {:.3f}'.format(epoch+1 , 300, loss.item()))

        print("Epoch: ", it, " Training Loss: ", cost_Extask1[-1],cost_Extask2[-1],cost_Extask3[-1], " Validation Loss: ", cost_val4[-1], cost_val5[-1], cost_val5[-1], " Test Loss: ", test_val4[-1], test_val5[-1], test_val5[-1])
                    
if __name__ == '__main__':
    main()