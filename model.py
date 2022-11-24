import torch
from torch import nn

class DigitModel(nn.Module):
    def __init__(self):

        super(DigitModel, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 32,kernel_size=4, stride=1),
            nn.ReLU()
        
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(32*7*7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        
        )

        self.head = nn.Linear(512, 7*7*(1+2*5)) #1+2*5

    def forward(self,x):
        x = self.convs(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.head(x)

        return x
        