import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, is_downsample=False):
        super(ResBlock, self).__init__()
        self.is_downsample = is_downsample

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        sample = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.is_downsample:
            sample = self.downsample(x)

        output += sample
        output = self.relu2(output)
        return output

class ResBottleneck(nn.Module):
    def __init__(self, previous_out, in_c, out_c, kernel_size, stride, is_downsample=False):
        super(ResBottleneck, self).__init__()
        self.is_downsample = is_downsample

        self.conv1 = nn.Conv2d(previous_out, in_c, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu3 = nn.ReLU()
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(previous_out, out_c, stride=stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        sample = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)

        if self.is_downsample:
            sample = self.downsample(x)

        output += sample
        output = self.relu3(output)
        return output


class ResNet(nn.Module):
    def __init__(self, mode, block, in_channels, block_def):
        super(ResNet, self).__init__()
        self.mode = mode
        self._in = 64
        self._out = 64
        self.previous_out = self._out

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self._out, kernel_size=7, stride=2),
            nn.BatchNorm2d(self._out),
            nn.ReLU()
        )

        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2x = self.generate_block(block, block_def[0], 3, 1)
        self.conv3x = self.generate_block(block, block_def[1], 3, 2, True)
        self.conv4x = self.generate_block(block, block_def[2], 3, 2, True)
        self.conv5x = self.generate_block(block, block_def[3], 3, 2, True)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.previous_out, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool(x)

        x = self.conv2x(x)
        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.conv5x(x)
        x = self.avgPool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def generate_block(self, block, layers, kernel_size=1, stride=1, downsample=False):
        layer = []
        is_downsample = downsample
        stride = stride

        match self.mode:
            case "18" | "34":
                for _ in range(layers):
                    layer.append(block(self._in, self._out, kernel_size, stride, is_downsample))
                    self._in = self._out
                    self.previous_out = self._out
                    if is_downsample:
                        is_downsample = False
                        stride = 1

                self._out *= 2
            case _:
                self._out = self._in*4
                for _ in range(layers):
                    layer.append(block(self.previous_out, self._in, self._out, kernel_size, stride, is_downsample))
                    self.previous_out = self._out

                    if is_downsample:
                        is_downsample = False
                        stride = 1

                self._in *= 2

        return nn.Sequential(*layer)
