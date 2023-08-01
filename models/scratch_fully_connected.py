import torch
import torch.nn as nn
import torchsummary 

class FullyConnected(nn.Module):
    def __init__(self, input_size = None, hidden_size = None, num_classes = None):
        super(FullyConnected, self).__init__()
        assert input_size is not None, "Input size cannot be None"
        assert hidden_size is not None, "Hidden size cannot be None"
        assert num_classes is not None, "Number of classes cannot be None"

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = self.flatten(x)
        # print(out.shape)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# flat_layer = nn.Flatten(start_dim=1, end_dim=-1)
# print(flat_layer(torch.randn(1, 3, 224, 224)).shape, 3*224*224)
# model = FullyConnected(input_size=3*224*224, hidden_size=1024, num_classes=7).to('cuda')
# torchsummary.summary(model, (3, 224, 224))