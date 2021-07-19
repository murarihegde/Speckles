import numpy as np
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self): 
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1,3, 5),         # (N, 1, 120, 120) -> (N,  6, 116, 116)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 116, 116) -> (N,  6, 58, 58)
            nn.Conv2d(3, 9, 5),        # (N, 6, 58, 58) -> (N, 12, 54, 54)  
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),   # (N,12, 54, 54) -> (N, 12, 27, 27)
            nn.Conv2d(9, 16, 3, stride=3), # (N, 12, 27, 27) -> (N, 16, 9, 9)
            nn.ReLU(),
            nn.AvgPool2d(3, 2)   # (N, 16, 9, 9) -> (n, 16, 4,4)
            # nn.Conv2d(16, 20, 3, stride=3), # (N, 16, 27, 27) -> (N, 20, 9, 9)
            # nn.ReLU()
        )

        self.fc_model = nn.Sequential(
            nn.Linear(16*4*4,120),         # (N, 1620) -> (N, 120)
            nn.Tanh(),
            nn.BatchNorm1d(120),
            nn.Linear(120,32),          # (N, 120) -> (N, 84)
            nn.Tanh(),
            nn.Linear(32,8),            # (N, 84)  -> (N, 10)
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        # print(x.shape)
        x = self.cnn_model(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc_model(x)
        # print(x.shape)
        return x