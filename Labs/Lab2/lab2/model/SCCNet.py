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
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=1, Nc=20, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=Nu, kernel_size=(C, Nt)),
            nn.BatchNorm2d(num_features=Nu),
            nn.Dropout(dropoutRate)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(Nu, 12)),
            nn.BatchNorm2d(num_features=Nc),
            SquareLayer(),
            nn.Dropout(dropoutRate)
        )

        self.avgPool = nn.AvgPool2d(kernel_size=(1, 62))
        self.fc = nn.Linear(in_features=120, out_features=numClasses)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)  # torch.Size([batch_size, 22, 22, 438])
        # print(x.shape)
        x = torch.permute(x, (0, 2, 1, 3))
        # print(x.shape)
        x = self.layer2(x)  # torch.Size([batch_size, 20, 1, 427])
        # print(x.shape)
        x = self.avgPool(x) # torch.Size([batch_size, 20, 1, 6])
        # print(x.shape)

        x = x.view(x.size(0), -1)   # torch.Size([batch_size, 120])
        x = self.fc(x)
        x = self.softmax(x)
        return x

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass