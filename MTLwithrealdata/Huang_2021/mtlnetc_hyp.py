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
# MMOE MODEL
input_size, feature_size = X_train.shape
mb_size1 = 5

class MTL_NetworkC(nn.Module):
    def __init__(self, trial, hidden_layer_size, task_layesr1, task_layesr2, num_inputs=feature_size, n_output_1=1, n_output_2=1,n_output_3=1):
        super(MTL_NetworkC, self).__init__()
        self.task_layesr1 = task_layesr1
        self.task_layesr2 = task_layesr2
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
    
    def forward(self, x):
        input_data = self.input_layer(x)
        out1 = self.output_layer(input_data)
        out2 = self.output_layer(input_data)
        out3 = self.hidden_layer_2(input_data)
        out4 = self.hidden_layer_3(input_data)
        
        return out1, out2, out3, out4
    
def objective(trial):
    # Hyperparameters
    task_layesr1 = trial.suggest_int('task_layesr1', 1, 10)
    task_layesr2 = trial.suggest_int('task_layesr2', 1, 10)
    hidden_layer_size =trial.suggest_int('hidden_layer_size', 10, 80)
    # Model
    model = MTL_NetworkC(trial,task_layesr1, task_layesr2, hidden_layer_size, num_inputs=feature_size, n_output_1=1, n_output_2=1,n_output_3=1)
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)  
    loss_func2 = nn.MSELoss()
    # Training
    for step in range(100):
        cost1tr_2 = []
        cost3tr_2 = []
        cost4tr_2 = []
        costtr_2 = []
        cost11tr = []
        cost12tr = []
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost11 = 0
        epoch_cost12 = 0
        epoch_cost3 = 0
        epoch_cost4 = 0
        num_minibatches = int(input_size / mb_size1)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, mb_size1)):
        # forward
            X_batch, Y1_batch, Y2_batch  = minibatch
            Y1_pred, Y2_pred, Y3_pred, Y4_pred = model(X_batch)
        # compute loss
            loss1 = loss_func2(Y1_pred, Y1_batch.view(-1,1))
            loss2 = loss_func2(Y2_pred, Y2_batch.view(-1,1))
            loss3 = loss_func2(Y3_pred, Y1_batch.view(-1,1))
            loss4 = loss_func2(Y4_pred, Y2_batch.view(-1,1))
            loss5 = loss1+ loss2
            loss = loss5 + loss3 + loss4
        # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
            val1_predict, val2_predict, val3_predict, val4_predict = model(X_valid)
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
    
    return cost3D_2  
    

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