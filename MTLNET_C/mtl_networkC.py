# MOdel
import torch.nn as nn
import torch.nn.functional as F


n_output_1 = 1
n_output_2 = 1
n_output_3 = 1

class MTL_NetworkC(nn.Module):
    def __init__(self, feature_size, hidden_layer_size):
        super(MTL_NetworkC, self).__init__()
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_layer_size
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
                            nn.Linear(hidden_layer_size, 5),
                            nn.Sigmoid(),
                            nn.Linear(5, 5),
                            nn.Sigmoid(),
                            nn.Linear(5, n_output_2))
        self.hidden_layer_3 = nn.Sequential(
                            nn.Linear(hidden_layer_size, 5),
                            nn.Sigmoid(),
                            nn.Linear(5, 5),
                            nn.Sigmoid(),
                            nn.Linear(5, n_output_3))                    
    
    def forward(self, x):
        input_data = self.input_layer(x)
        out1 = self.output_layer(input_data)
        out2 = self.output_layer(input_data)
        out3 = self.hidden_layer_2(input_data)
        out4 = self.hidden_layer_3(input_data)
        
        return out1, out2, out3, out4