import torch
from torch import nn


class NeuralClassifier(nn.Module):

    def __init__(self, num_features: int):
        # batch_size by timepoints by bands by bins
        super().__init__()
        self.flatten = nn.Flatten(0, 1)
        self.conv2d = nn.Conv2d(1, 64, 3)
        self.max2d = nn.MaxPool2d(3) # pooling layer
        
        self.lstm = nn.LSTM(64, 64, batch_first = True)
        self.dropout = nn.Dropout(p = 0.3)
        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 32)
        self.relu = nn.Relu()
        self.linear3 = nn.Linear(32, 2)
        self.softmax = nn.Softmax()

    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        size = data.size()

        x = self.flatten(data)
        x = self.conv2d(data)
        x = self.max2d(x)

        x = x.unflatten(0, (size[0], size[1]))
        
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.softmax(self.linear3(x))
        return x