# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelAttention, SpatialAttention, CBAM is referenced from the following site:
# https://arxiv.org/pdf/1807.06521
# https://github.com/luuuyi/CBAM.PyTorch/tree/master
# https://blog.csdn.net/weixin_41790863/article/details/123413303
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16) -> None:
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

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
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

class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UnetConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.unetconv = UnetConv(in_channels, out_channels)
        self.cbam = CBAM(out_channels)
    
    def forward(self, x, prev_x):
        x = self.upconv(x)

        diffY = x.size()[2] - prev_x.size()[2]
        diffX = x.size()[3] - prev_x.size()[3]
        prev_x = F.pad(prev_x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([prev_x, x], dim=1)
        
        x = self.unetconv(x)
        x = self.cbam(x)
        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_downsample=False) -> None:
        super(ResBlock, self).__init__()
        self.is_downsample = is_downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        if self.is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        sample = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.is_downsample:
            sample = self.downsample(x)

        out += sample
        out = self.relu2(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, total_blocks, is_downsample=True) -> None:
        super(EncoderBlock, self).__init__()

        blocks = []
        blocks.append(ResBlock(in_channels, out_channels, kernel_size=3, stride=1+1*is_downsample, is_downsample=is_downsample))
        for _ in range(1, total_blocks):
            blocks.append(ResBlock(out_channels, out_channels, kernel_size=3, stride=1))
        
        self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer(x)
        return x


class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ResNet34_UNet, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.enc1 = EncoderBlock(64, 64, total_blocks=3, is_downsample=False)
        self.enc2 = EncoderBlock(64, 128, total_blocks=4)
        self.enc3 = EncoderBlock(128, 256, total_blocks=6)
        self.enc4 = EncoderBlock(256, 512, total_blocks=3)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv = UnetConv(32, 32)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            UnetConv(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.upsample1(x)
        x = self.upconv(x)

        x = self.out_conv(x)
        return x