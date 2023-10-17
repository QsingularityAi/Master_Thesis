import torch
import torch.nn as nn
import torch.optim as optim
from data import data_preparation
import optuna
from optuna.trial import TrialState
from utils import random_mini_batches

mb_size1 = 100
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []
# Define Data pre-peraration
X_train, Y1_train, Y2_train, X_valid, Y1_valid, Y2_valid = data_preparation(num_feature=100, rho=0.8, num_row = 10000)
input_size, feature_size = X_train.shape

import torch
import torch.nn as nn

class polymerinformatic(nn.Module):
    def __init__(self, trial, hidden_layer1, hidden_layer2, hidden_layer3, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_layer1= hidden_layer1
        self.hidden_layer2= hidden_layer2
        self.hidden_layer3= hidden_layer3
        self.fc = nn.Linear(feature_size, hidden_layer1)
        self.fc1 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc2 = nn.Linear(hidden_layer2, hidden_layer3)
        self.fc3 = nn.Linear(hidden_layer3, 1)
        self.fc4 = nn.Linear(hidden_layer3, 1)
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
        x = self.relu(x)
        x = self.dropout(x)
        x2 = self.fc4(x)
        return x1, x2
    
def objective(trial):
    print()
    print("Training...")
    # Define Data pre-peraration
    # Define Hyperparameters    
    hidden_layer1 = trial.suggest_int('hidden_layer1', 0, 224)
    hidden_layer2 = trial.suggest_int('hidden_layer2', 0, 160)
    hidden_layer3 = trial.suggest_int('hidden_layer3', 1, 36)
    
    # Model
    model = polymerinformatic(trial, hidden_layer1, hidden_layer2, hidden_layer3, feature_size)
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
              
