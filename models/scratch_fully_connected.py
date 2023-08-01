import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, input_size = None, hidden_size = None, num_classes = None):
        super(FullyConnected, self).__init__()
        assert input_size is not None, "Input size cannot be None"
        assert hidden_size is not None, "Hidden size cannot be None"
        assert num_classes is not None, "Number of classes cannot be None"

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out