# Implement your UNet model here

# assert False, "Not implemented yet!"

import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size) -> None:
        super(Conv2dBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_feature, out_channels=out_feature, kernel_size=kernel_size)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_features, layers) -> None:
        super(UNet, self).__init__()
        
        self.in_features = in_features
        self.conv1 = Conv2dBlock(1, in_features, kernel_size=3)
        
        up_layers = []
        for i in range(1, layers):
            if i < int(layers/2):
                up_layers.append(Conv2dBlock(self.in_features, self.in_features*2, kernel_size=3))
                up_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.in_features = self.in_features*2
            elif i > int(layers/2):
                up_layers.append(nn.Conv2d(self.in_features, self.in_features, kernel_size=2))
                up_layers.append(Conv2dBlock(self.in_features, int(self.in_features*0.5), kernel_size=2))                
                self.in_features = int(self.in_features*0.5)
            else:
                up_layers.append(Conv2dBlock(self.in_features, self.in_features*2, kernel_size=3))
                self.in_features = self.in_features*2

        self.net = nn.Sequential(*up_layers)
        self.classifier = nn.Conv2d(self.in_features, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.net(x)
        x = self.classifier(x)

        return x

