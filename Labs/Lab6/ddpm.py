from dataloader import iClevrLoader
from evaluator import evaluation_model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


# 利用json檔中提供的標籤，從noise裡畫出標籤中的物品
# 訓練的話，label就是真實圖片，data則是noise和標籤
# loss就要算真實圖片和預測圖片的差異(在相同的timestep下，timestep越小，noise越多)
class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = UNet2DModel(
            sample_size=self.args.sample_size,
            in_channels=self.args.in_channels,      # 需要吃json的標籤和noise(shape和真實圖片相同)當作input
            out_channels=self.args.out_channels,    # output預測的圖片
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
        b, c, w, h = noise.shape
        
        # reshape label to fit the input of model
        reshaped_label = label.view(b, label.shape[1], 1, 1).expand(b, label.shape[1], w, h)
        sample = torch.cat((noise, reshaped_label), 1)

        return self.model(sample=sample, timestep=timestep).sample

    def train(self):
        losses = []

        for epoch in range(self.args.n_epoch):
            train_dataloader = DataLoader(iClevrLoader(self.args.root, self.args.mode), batch_size=self.args.batch_size, shuffle=True)

            for img, label in tqdm(train_dataloader):
                    img = img.to(self.args.device)      # real img
                    label = label.to(self.args.device)  # label of wanted object
                    timestep = torch.randint(0, self.args.num_train_timesteps-1, (img.shape[0],)).long().to(self.args.device)
                    noise = torch.randn_like(img).to(self.args.device)
                    noisy_img = self.scheduler.add_noise(img, noise, timestep)   # real img in same timestep of pred img from noise
                    
                    pred = self.model(noise, label, timestep)

                    loss = self.criterion(pred, noisy_img)  # loss of real img and pred img (under same timestep)

                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)
                    self.optim.step()

                    losses.append(loss.item())

            # average of last 100 loss values
            avg_loss = sum(losses[-100:]) / 100.0
            print(f'Epoch {epoch}: Average of last 100 loss values = {avg_loss}')

            # draw losses and save model
            plt.plot(losses)
            self.save_model(os.path.join(self.args.save_root, f'checkpoint{avg_loss}.pth'))

    def test(self):
        test_dataloader = DataLoader(iClevrLoader(self.args.root, self.args.mode), batch_size=self.args.batch_size, shuffle=False)

        self.model.eval()
        for label in test_dataloader:
            sample = torch.randn(len(label), self.args.out_channels, self.args.sample_size, self.args.sample_size).to(self.args.device)
            label = label.to(self.args.device)

            for i, t in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
                with torch.no_grad():
                    residual = self.model(sample, label, t)

                sample = self.scheduler.step(residual, t, sample).prev_sample

            # calculate accuracy
            evaluator = evaluation_model(self.args.ckpt_path)
            acc = evaluator.eval(sample, label)
            print(f'Test accruacy: {acc}')

            # plot images
            # grid_img = make_grid(sample, 8)
            # save_image()

    def save_model(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict()
        }, path)
        print(f'> Save model to {path}...')


def main(args):
    model = DDPM(args).to(args.device)

    if args.test_only:
        model.test()
    else:
        model.train()


if __name__ == '__main__':
    ## arguments ##
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device',         type=str,           default='cuda')
    parser.add_argument('--root',           type=str,           default='D:/DL Course/DLCourse/Labs/Lab6')
    parser.add_argument('--save_root',      type=str,           default='D:/DL Course/DLCourse/Labs/Lab6/checkpoints')
    parser.add_argument('--ckpt_path',      type=str,           default='D:/DL Course/DLCourse/Labs/Lab6/checkpoints/checkpoint.pth')
    parser.add_argument('--test_only',      action='store_true')
    parser.add_argument('--mode',           type=str,           default='train')
    # hyperparameters
    parser.add_argument('--lr',             type=float,         default=.001)
    parser.add_argument('--n_epoch',        type=int,           default=1)
    parser.add_argument('--batch_size',     type=int,           default=256)
    parser.add_argument('--in_channels',    type=int,           default=27)  # rgb 3 channels + 24 class of objects
    parser.add_argument('--out_channels',   type=int,           default=3)  # predicted picture (same size and channels of origin pictures)
    parser.add_argument('--sample_size',    type=int,           default=64)  # origin pictures of size 64*64
    parser.add_argument('--num_train_timesteps',    type=int,   default=1000)
    parser.add_argument('--beta_schedule',  type=str,           default='squaredcos_cap_v2')
    args = parser.parse_args()

    ## main ##
    main(args)