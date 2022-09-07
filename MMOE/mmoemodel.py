
import torch
import torch.nn as nn

class MMOE(nn.Module):
    def __init__(self, num_experts, num_tasks, num_inputs, hidden_expert_neurons, hidden_expert_neurons2, hidden_tower_neurons, drop_rate=0.7):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
        self.hidden_expert_neurons = hidden_expert_neurons
        self.hidden_expert_neurons2 = hidden_expert_neurons2
        self.hidden_tower_neurons = hidden_tower_neurons

        ## Experts Layer

        for i in range(num_experts):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(num_inputs, hidden_expert_neurons),
                nn.ReLU(),
                nn.Linear(hidden_expert_neurons, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_expert_neurons2),
                nn.ReLU(),
                #nn.Dropout(p=drop_rate)
            ))

        ## Gates Layer
        for i in range(num_tasks): # number of towers, fixed to 2.
            setattr(self, 'gate'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_experts),
                nn.Softmax(dim=1)
            ))

        ## Towers Layer
        for i in range(num_tasks):
            setattr(self, 'tower'+str(i), nn.Linear(hidden_expert_neurons2, hidden_tower_neurons))

    def forward(self, xx):
        a = xx.shape[0]
        ## experts output
        out_experts = torch.zeros(self.num_experts, a, self.hidden_expert_neurons2)
        for i in range(self.num_experts):
            out_experts[i] = getattr(self, 'expert'+str(i))(xx)

        ## gates and weights
        input_towers = torch.zeros(self.num_tasks, a, self.hidden_expert_neurons2)
        for i in range(self.num_tasks):
            gate = getattr(self, 'gate'+str(i))(xx)
            for j in range(self.num_experts):
                input_towers[i] += gate[:,j].unsqueeze(dim=1)*out_experts[j]

        ## towers output
        out_towers = torch.zeros(self.num_tasks, a , self.hidden_tower_neurons)
        for i in range(self.num_tasks):
            out_towers[i] = getattr(self, 'tower'+str(i))(input_towers[i])
        output = torch.sigmoid(out_towers)

        # return out_towers
        return output[0], output[1]