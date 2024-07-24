# Implement your ResNet34_UNet model here

# assert False, "Not implemented yet!"

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=2) -> None:
        super(ChannelAttention, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channels, channels//ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels//ratio, channels, kernel_size=1)
        )
        self.sigmid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.shared_MLP(self.maxpool(x))
        avg_out = self.shared_MLP(self.avgpool(x))
        out = self.sigmid(max_out + avg_out)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out
    
class CBAM(nn.Module):
    def __init__(self, channels) -> None:
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class Res34Block(nn.Module):
    def __init__(self, in_feature, out_feature, up_sample) -> None:
        super(Res34Block, self).__init__()

        self.is_up_sample = up_sample

        self.conv1 = nn.Conv2d(in_feature, in_feature, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_feature)
        self.conv2 = nn.Conv2d(in_feature, out_feature, kernel_size=3)
        if self.is_up_sample:
            self.cbam = CBAM(out_feature)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_feature)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.conv2(out)

        if self.is_up_sample:
            out = self.cbam(out)

        out += x 
        out = self.relu2(out)
        out = self.bn2(out)
        return out

class UNet_ResNet34(nn.Module):
    def __init__(self, in_features, unet_layers) -> None:
        super(UNet_ResNet34, self).__init__()

        self.in_features = in_features
        self.conv1 = nn.Conv2d(3, in_features, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3)

        net_layers = []
        for i in range(unet_layers):
            if i < int(unet_layers/2):
                net_layers.append(Res34Block(self.in_features, self.in_features*2, up_sample=False))
                self.in_features *= 2
            elif i > int(unet_layers/2):
                net_layers.append(Res34Block(self.in_features, int(self.in_features*0.5), up_sample=True))
                self.in_features = int(self.in_features*0.5)
            else:
                net_layers.append(Res34Block(self.in_features, int(self.in_features*0.5), up_sample=False))
                self.in_features = int(self.in_features*0.5)

        self.net = nn.Sequential(*net_layers)
        self.conv3 = nn.Sequential(
            Res34Block(self.in_features, self.in_features, up_sample=True),
            Res34Block(self.in_features, self.in_features, up_sample=True),
            Res34Block(self.in_features, 1, up_sample=False),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(2)
        x = self.net(x)
        x = self.conv3(x)

        return x