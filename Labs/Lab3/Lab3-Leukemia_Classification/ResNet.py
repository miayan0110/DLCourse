import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResNet(nn.Module):
    def __init__(self, block_def):
        super().__init__()
        self.blockOutput = []
        self.activation = nn.ReLU()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.maxPool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxPool(x)

class ResNet18(ResNet):
    def __init__(self, block_def):
        super().__init__(block_def)

        self.block2 = nn.ModuleList([nn.Conv2d(3, 64, kernel_size=3) for layer in range(block_def[0])])
        self.block3= nn.ModuleList([nn.Conv2d(3, 128, kernel_size=3) for layer in range(block_def[1])])
        self.block4 = nn.ModuleList([nn.Conv2d(3, 256, kernel_size=3) for layer in range(block_def[2])])
        self.block5 = nn.ModuleList([nn.Conv2d(3, 512, kernel_size=3) for layer in range(block_def[3])])

    def generate_block(self, layers, out_channels, kernel_size):
        layer = [nn.Conv2d(3, out_channels, kernel_size=kernel_size) for layer in range(layers)]
        layer.append()