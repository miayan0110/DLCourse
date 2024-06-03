from dataloader import iClevrLoader
import argparse
import torch.nn as nn
import torch.optim as optim
from diffusers import UNet2DModel
from diffusers import DDPMScheduler


class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ],
            up_block_types=[
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ],
            block_out_channels=[64, 128, 128],
            layers_per_block=2
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        self.optim = optim.Adam(self.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

    def forward(self):
        return self.model(sample=, timestep=)



def main(args):
    dataset = iClevrLoader(args.root, args.mode)
    img, label = dataset[0]
    print(img.shape)

if __name__ == '__main__':
    ## arguments ##
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-r', '--root', default='D:/DL Course/DLCourse/Labs/Lab6')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test_only', action='store_true')
    # hyperparameters
    parser.add_argument('--lr', default=.0005, type=float)
    args = parser.parse_args()

    ## main ##
    main(args)