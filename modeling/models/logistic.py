import torch
from torch import nn


class LogisticClassifier(nn.Module):

    def __init__(self, num_features: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=num_features, out_features=2)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self.flatten(data)
        x = self.linear(x)
        return x