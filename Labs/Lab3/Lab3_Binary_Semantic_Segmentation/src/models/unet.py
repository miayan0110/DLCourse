# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

# referenced from https://github.com/milesial/Pytorch-UNet
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(Conv2dBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DownsampleBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBlock(in_channels, out_channels, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsampleBolck(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UpsampleBolck, self).__init__()

        self.upconv =  nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = Conv2dBlock(in_channels, out_channels, kernel_size=3)

    def forward(self, x, prev_x):
        x = self.upconv(x)

        diffY = prev_x.size()[2] - x.size()[2]
        diffX = prev_x.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([prev_x, x], dim=1)

        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels) -> None:
        super(UNet, self).__init__()
        
        self.in_conv = Conv2dBlock(in_channels, 64, kernel_size=3)

        self.down1 = DownsampleBlock(64, 128)
        self.down2 = DownsampleBlock(128, 256)
        self.down3 = DownsampleBlock(256, 512)
        self.down4 = DownsampleBlock(512, 1024)
        self.up1 = UpsampleBolck(1024, 512)
        self.up2 = UpsampleBolck(512, 256)
        self.up3 = UpsampleBolck(256, 128)
        self.up4 = UpsampleBolck(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        x = self.sigmoid(x)

        return x

