import torch
import torch.nn as nn

class polymerinformatic(nn.Module):
    def __init__(self, feature_size, hidden_layer1, hidden_layer2, hidden_layer3):
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