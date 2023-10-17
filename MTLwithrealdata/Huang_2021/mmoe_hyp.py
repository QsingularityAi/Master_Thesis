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
mb_size1 = 10
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []

class MMOE(nn.Module):
    def __init__(self,trial, num_experts, num_neurons_expert, hidden_neu_expert, hidden_tower_neu, num_neurons_expert2, num_tasks,num_neurons_tower, num_inputs=feature_size):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        # self.num_inputs = num_inputs
        self.hidden_neu_expert = hidden_neu_expert
        self.num_neurons_expert = num_neurons_expert
        self.num_neurons_expert2 = num_neurons_expert2
        self.hidden_tower_neu = hidden_tower_neu
        self.num_neurons_tower = num_neurons_tower
        ## Experts
        for i in range(num_experts):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_neurons_expert),
                nn.ReLU(),
                nn.Linear(num_neurons_expert, hidden_neu_expert),
                nn.LeakyReLU(),
                nn.Linear(hidden_neu_expert, num_neurons_expert2),
                nn.ReLU()
            ))
        ## Gates
        for i in range(num_tasks): # number of towers, fixed to 2.
            setattr(self, 'gate'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_experts),
                nn.Softmax(dim=1)
            ))
        ## Towers
        for i in range(num_tasks):
            setattr(self, 'tower'+str(i), nn.Sequential(
                nn.Linear(num_neurons_expert2, hidden_tower_neu),
                # nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(hidden_tower_neu, num_neurons_tower),
            ))#nn.Linear(num_neurons_expert2, num_neurons_tower))

    def forward(self, xv):
        bs = xv.shape[0]
        ## experts
        out_experts = torch.zeros(self.num_experts, bs, self.num_neurons_expert2)
        for i in range(self.num_experts):
            out_experts[i] = getattr(self, 'expert'+str(i))(xv)
        ## gates and weighted opinions
        input_towers = torch.zeros(self.num_tasks, bs, self.num_neurons_expert2)
        for i in range(self.num_tasks):
            gate = getattr(self, 'gate'+str(i))(xv)
            for j in range(self.num_experts):
                input_towers[i] += gate[:,j].unsqueeze(dim=1)*out_experts[j]
        ## towers
        out_towers = torch.zeros(self.num_tasks, bs, self.num_neurons_tower)
        for i in range(self.num_tasks):
            out_towers[i] = getattr(self, 'tower'+str(i))(input_towers[i])
        output = torch.sigmoid(out_towers)
        # return out_towers
        return output[0], output[1]
    
def objective(trial):
    # Hyperparameters
    num_experts = trial.suggest_int('num_experts', 1, 5)
    hidden_neu_expert = trial.suggest_int('hidden_neu_expert', 1, 30)
    hidden_tower_neu = trial.suggest_int('hidden_tower_neu', 1, 10)
    num_neurons_expert = trial.suggest_int('num_neurons_expert', 1, 50)
    num_neurons_expert2 = trial.suggest_int('num_neurons_expert2', 1, 20)
    # Model
    model = MMOE(trial, num_experts, num_neurons_expert, hidden_neu_expert, hidden_tower_neu, num_neurons_expert2, num_tasks=2,num_neurons_tower=1, num_inputs=feature_size)
    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr) 
    loss_func1 = nn.MSELoss()
    # Training
    for step in range(100):
        costtr = []
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        num_minibatches = int(input_size / mb_size1)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, mb_size1)):
        # forward
            X_batch, Y1_batch, Y2_batch  = minibatch
            predict = model(X_batch)
        # compute loss
            loss1 = loss_func1(predict[0], Y1_batch.view(-1,1))
            loss2 = loss_func1(predict[1], Y2_batch.view(-1,1))
            loss = loss1 + loss2
        # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
        costtr.append(torch.mean(epoch_cost))
        cost1tr.append(torch.mean(epoch_cost1))
        cost2tr.append(torch.mean(epoch_cost2))
        # Evaluation
        model.eval()
        with torch.no_grad():
            predict = model(X_valid)
            l1D = loss_func1(predict[0], Y1_valid.view(-1,1))
            l2D = loss_func1(predict[1], Y2_valid.view(-1,1))
            cost1D.append(l1D)
            cost2D.append(l2D)
            costD.append(l1D+l2D)
    return costtr  
    

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
    