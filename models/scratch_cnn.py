import torch
import torch.nn as nn

class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=512, out_features=7)
        self.softmax = nn.Softmax(dim=1)

        self.model = nn.Sequential(
            self.conv1, self.relu1, self.pool,
            self.conv2, self.relu2, self.pool,
            self.conv3, self.relu3, self.pool,
            self.fc1, self.relu4,
            self.fc2, self.relu5,
            self.fc3, self.softmax
        )

    def forward(self, x):
        return self.model(x)
