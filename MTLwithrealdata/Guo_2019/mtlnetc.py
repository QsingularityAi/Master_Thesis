import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from utils import random_mini_batches

from data import data_prepration


X_train, Y1_train, Y2_train, Y3_train, X_valid, Y1_valid, Y2_valid, Y3_valid = data_prepration()

num_inputs = X_train.shape[1]
# MMOE MODEL
input_size, feature_size = X_train.shape
mb_size1 = 50

class MTL_NetworkC(nn.Module):
    def __init__(self, trial, hidden_layer_size, task_layesr1, task_layesr2, task_layesr3, num_inputs=feature_size, n_output_1=1, n_output_2=1,n_output_3=1, n_output_4=1):
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
    
def objective(trial):
    # Hyperparameters
    task_layesr1 = trial.suggest_int('task_layesr1', 1, 15)
    task_layesr2 = trial.suggest_int('task_layesr2', 1, 15)
    task_layesr3 = trial.suggest_int('task_layesr3', 1, 15)
    hidden_layer_size =trial.suggest_int('hidden_layer_size', 10, 100)
    # Model
    model = MTL_NetworkC(trial,task_layesr1, task_layesr2, task_layesr3, hidden_layer_size, num_inputs=feature_size, n_output_1=1, n_output_2=1,n_output_3=1)
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr) 
    loss_func2 = nn.MSELoss()
    # Training
    for step in range(100):
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
        Total_cost = 0
        Combine_cost= 0
        Task1_cost = 0
        Task2_cost = 0
        Task3_cost = 0
        Task1_With_Exlayer = 0
        Task2_With_Exlayer = 0
        Task3_With_Exlayer = 0
        num_minibatches = int(input_size / mb_size1)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, Y3_train, mb_size1)):
        # forward
            X_batch, Y1_batch, Y2_batch, Y3_batch = minibatch
            Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred = model(X_batch)
        # compute loss
            loss1 = loss_func2(Y1_pred, Y1_batch.view(-1,1))
            loss2 = loss_func2(Y2_pred, Y2_batch.view(-1,1))
            loss3 = loss_func2(Y3_pred, Y3_batch.view(-1,1))
            loss4 = loss_func2(Y4_pred, Y1_batch.view(-1,1))
            loss5 = loss_func2(Y5_pred, Y2_batch.view(-1,1))
            loss6 = loss_func2(Y6_pred, Y3_batch.view(-1,1))
            loss7 = loss1+ loss2 + loss3
            loss = loss7 + loss4 + loss5 + loss6
        # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        # Evaluation
        cost1D_2 = []
        cost2D_2 = []
        cost3D_2 = []
        cost4D_2 = []
        cost11D = []
        cost12D = []
        costD_2 = []
        model.eval()
        with torch.no_grad():
            val1_predict, val2_predict, val3_predict, val4_predict, val5_predict, val6_predict = model(X_valid)
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
    
    return cost_Total_val  
    

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))