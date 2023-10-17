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
# MMOE MODEL
input_size, feature_size = X_train.shape
mb_size1 = 200

class NeuralNetwork(nn.Module):
    def __init__(self,trial,hidden_nuron1, hidden_nuron2, hidden_nuron3, num_inputs=feature_size ):
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
    
def objective(trial):
    # Hyperparameters
    hidden_nuron1 = trial.suggest_int('hidden_nuron1', 1, 180)
    hidden_nuron2 = trial.suggest_int('hidden_nuron2', 1, 64)
    hidden_nuron3 = trial.suggest_int('hidden_nuron3', 1, 32)
    # Model
    model = NeuralNetwork(trial,hidden_nuron1, hidden_nuron2, hidden_nuron3, num_inputs=feature_size)
    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay) 
    loss_func1 = nn.MSELoss()
    # Training
    for step in range(100):
        costtr = []
        cost1tr = []
        cost2tr = []
        cost3tr = []
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        epoch_cost3 = 0
        num_minibatches = int(input_size / mb_size1)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, Y3_train, mb_size1)):
        # forward
            X_batch, Y1_batch, Y2_batch, Y3_batch  = minibatch
            predict = model(X_batch)
        # compute loss
            loss1 = loss_func1(predict[0], Y1_batch.view(-1,1))
            loss2 = loss_func1(predict[1], Y2_batch.view(-1,1))
            loss3 = loss_func1(predict[2], Y3_batch.view(-1,1))
            loss = loss1 + loss2 + loss3
        # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
            epoch_cost3 = epoch_cost3 + (loss3 / num_minibatches)
        costtr.append(torch.mean(epoch_cost))
        cost1tr.append(torch.mean(epoch_cost1))
        cost2tr.append(torch.mean(epoch_cost2))
        cost3tr.append(torch.mean(epoch_cost3))
        # Evaluation
        costD = []
        cost1D = []
        cost2D = []
        cost3D = []
        model.eval()
        with torch.no_grad():
            predict = model(X_valid)
            l1D = loss_func1(predict[0], Y1_valid.view(-1,1))
            l2D = loss_func1(predict[1], Y2_valid.view(-1,1))
            l3D = loss_func1(predict[2], Y3_valid.view(-1,1))
            cost1D.append(l1D)
            cost2D.append(l2D)
            cost3D.append(l3D)
            costD.append(l1D+l2D+l3D)
    return costD  
    

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