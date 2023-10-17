# MOdel
import torch.nn as nn
import torch.nn.functional as F

class MTL_NetworkC(nn.Module):
    def __init__(self, feature_size, hidden_layer_size, task_layer1, task_layer2, n_output_1, n_output_2, n_output_3):
        super(MTL_NetworkC, self).__init__()
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_layer_size
        self.task_layer1 = task_layer1
        self.task_layer2 = task_layer2
        self.n_output_1 = n_output_1
        self.n_output_2 = n_output_2
        self.n_output_3 = n_output_3
        self.input_layer = nn.Sequential(
                           nn.Linear(feature_size, hidden_layer_size),
                           nn.Sigmoid()
                           )
        self.hidden_layer_1 = nn.Sequential(
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.Sigmoid(),
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.Sigmoid()
                            )
        self.output_layer = nn.Sequential(nn.Linear(hidden_layer_size, n_output_1),
                            nn.Sigmoid())
        self.hidden_layer_2 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layer1),
                            nn.Sigmoid(),
                            nn.Linear(task_layer1, task_layer1),
                            nn.Sigmoid(),
                            nn.Linear(task_layer1, n_output_2))
        self.hidden_layer_3 = nn.Sequential(
                            nn.Linear(hidden_layer_size, task_layer2),
                            nn.Sigmoid(),
                            nn.Linear(task_layer2, task_layer2),
                            nn.Sigmoid(),
                            nn.Linear(task_layer2, n_output_3))                    
    
    def forward(self, x):
        input_data = self.input_layer(x)
        out1 = self.output_layer(input_data)
        out2 = self.output_layer(input_data)
        out3 = self.hidden_layer_2(input_data)
        out4 = self.hidden_layer_3(input_data)
        
        return out1, out2, out3, out4