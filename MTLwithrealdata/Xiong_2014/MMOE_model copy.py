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
mb_size = 6
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []
costts = []

class MMOE(nn.Module):
    def __init__(self, num_experts, num_tasks, num_inputs, num_neurons_expert, num_neurons_expert2, num_neurons_tower):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
        self.num_neurons_expert = num_neurons_expert
        self.num_neurons_expert2 = num_neurons_expert2
        self.num_neurons_tower = num_neurons_tower
        ## Experts
        for i in range(num_experts):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_neurons_expert),
                nn.LeakyReLU(),
                nn.Linear(num_neurons_expert, 5),
                nn.LeakyReLU(),
                nn.Linear(5, num_neurons_expert2),
                nn.LeakyReLU(),
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
                nn.Linear(num_neurons_expert2, 9),
                # nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(9, num_neurons_tower),
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
    
    
MTL1 = MMOE(num_experts=5, num_tasks=2, num_inputs=num_inputs, num_neurons_expert=6,num_neurons_expert2 =7, num_neurons_tower=1)
gamma1 = 1
step_size1 = 100
optimizer1 = torch.optim.RMSprop(MTL1.parameters(), lr=0.0014637044957925424, weight_decay=3.563015118407307e-06)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size1, gamma=gamma1)
loss_func1 = nn.MSELoss()


def main():
    # Train model
    print()
    print("Training...")
    
    for it in range(epoch1):
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        num_minibatches = int(input_size / mb_size)
        for minibatch in (random_mini_batches(X_train, Y1_train, Y2_train, mb_size)):
            X_batch, Y1_batch, Y2_batch  = minibatch
            predict = MTL1(X_batch)
            loss1 = loss_func1(predict[0], Y1_batch.view(-1,1))
            loss2 = loss_func1(predict[1], Y2_batch.view(-1,1))
            
            loss = loss1 + loss2
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            scheduler.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (loss1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
            
        costtr.append(torch.mean(epoch_cost))
        cost1tr.append(torch.mean(epoch_cost1))
        cost2tr.append(torch.mean(epoch_cost2))
        with torch.no_grad():
            predict = MTL1(X_valid)
            l1D = loss_func1(predict[0], Y1_valid.view(-1,1))
            l2D = loss_func1(predict[1], Y2_valid.view(-1,1))
            cost1D.append(l1D)
            cost2D.append(l2D)
            costD.append(l1D+l2D)
        MTL1.eval()
        with torch.no_grad():
            predict = MTL1(X_test)    
            l1ts = loss_func1(predict[0], Y1_test.view(-1,1))
            l2ts = loss_func1(predict[1], Y2_test.view(-1,1))
            cost1ts.append(l1ts)
            cost2ts.append(l2ts)
            costts.append(l1ts+l2ts)
        print("Epoch: ", it, " Training Loss: ", cost1tr[-1],cost2tr[-1], " Validation Loss: ", cost1D[-1], cost2D[-1], "test Loss:", cost1ts[-1],cost2ts[-1])
                    
if __name__ == '__main__':
    main()        