
import torch
import torch.nn as nn

class MMOE(nn.Module):
    def __init__(self, num_experts, num_neurons_expert, hidden_neu_expert, hidden_tower_neu, num_neurons_expert2, num_tasks,num_neurons_tower, num_inputs):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
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