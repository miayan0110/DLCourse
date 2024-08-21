import torch
from torch import nn
from diffusers import UNet2DModel


class ConditionDDPM(nn.Module):
    def __init__(self, img_channel, num_class) -> None:
        super().__init__()
        self.num_class = num_class
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=img_channel + num_class,    # concat label classes and image channels
            out_channels=img_channel,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            )
        )

    def forward(self, x, labels, t):
        b, c, w, h = x.shape

        # change shape of labels to fit the images
        labels = labels.view(b, self.num_class, 1, 1).expand(b, self.num_class, w, h)

        # concat labels into img to make inputs
        input = torch.cat((x, labels), dim=1)   # shape (b, c+num_class, w, h)
        return self.model(input, t).sample
