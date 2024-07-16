# implement SCCNet model

import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.square(x)

class SCCNet(nn.Module):
    def __init__(self, numClasses=0, timeSample=0, Nu=0, C=0, Nc=0, Nt=0, dropoutRate=0):
        super(SCCNet, self).__init__()

        self.timeSample = timeSample

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=Nu, kernel_size=(C, Nt), padding=(0, Nt//2)),
            nn.BatchNorm2d(num_features=Nu),
            SquareLayer(),
            nn.Dropout(dropoutRate)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(Nu, 12), padding=(0, 6)),
            nn.BatchNorm2d(num_features=Nc),
            SquareLayer(),
            nn.Dropout(dropoutRate)
        )

        self.avgPool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
        self.fc = nn.Linear(in_features=704, out_features=numClasses)

    def forward(self, x):
        x = self.layer1(x)  # torch.Size([1, 22, 22, 438])
        x = self.layer2(x)  # torch.Size([1, 22, 1, 439])
        x = self.avgPool(x) # torch.Size([1, 22, 1, 32])

        x = x.view(x.size(0), -1)   # torch.Size([1, 704])
        x = self.fc(x)
        return x

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass