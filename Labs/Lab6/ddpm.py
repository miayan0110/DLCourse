from dataloader import iClevrLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import tqdm


class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = UNet2DModel(
            sample_size=self.args.sample_size,
            in_channels=self.args.in_channels, # 需要吃json的標籤和noise(shape和真實圖片相同)當作input
            out_channels=self.args.out_channels, # output預測的圖片
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

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.args.num_train_timesteps, 
            beta_schedule=self.args.beta_schedule
        )

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss()

    def forward(self, noise, label, timestep):

        return self.model(sample=, timestep=timestep).sample

    def train(self):
        losses = []
        self.model.to(self.args.devce)

        for epoch in range(self.args.episode):
            train_dataloader = DataLoader(iClevrLoader(self.args.root, self.args.mode), batch_size=self.args.batch_size, shuffle=True)

            for img, label in tqdm(train_dataloader):
                    img = img.to(self.args.device)
                    label = label.to(self.args.device)
                    timestep = torch.randint(0, self.args.num_train_timesteps-1).long().to(self.args.device)
                    noise = torch.randn_like(img).to(self.args.device)
                    
                    pred = self.model(noise, label, timestep)

                    loss = self.criterion(pred, img)

    def test(self):
        test_dataloader = DataLoader(iClevrLoader(self.args.root, self.args.mode), batch_size=self.args.batch_size, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            # 利用json檔中提供的標籤，從noise裡畫出標籤中的物品
            # 訓練的話，label就是真實圖片，data則是noise和標籤
            # loss就要算真實圖片和預測圖片的差異(在相同的timestep下，timestep越小，noise越多)
            for label in tqdm(test_dataloader):
                label = label.to(self.args.device)
                noise = torch.randn(len(label), self.args.out_channels, self.args.size, self.args.size).to(self.args.device)
                pred = self.model(noise, label, self.args.num_train_timesteps)



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
    parser.add_argument('--episode', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--in_channels', default=27, type=int)  # rgb 3 channels + 24 class of objects
    parser.add_argument('--out_channels', default=3, type=int)  # predicted picture (same size and channels of origin pictures)
    parser.add_argument('--sample_size', default=64, type=int)  # origin pictures of size 64*64
    parser.add_argument('--num_train_timesteps', default=1000, type=int)
    parser.add_argument('--beta_schedule', default='squaredcos_cap_v2')
    args = parser.parse_args()

    ## main ##
    main(args)