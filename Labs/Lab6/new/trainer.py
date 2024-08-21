import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm
import numpy as np

from dataloader import IClevrDataSet
from model import ConditionDDPM
from evaluator import evaluation_model

#############   Trainer   #############
class DDPMTrainer:
    def __init__(self, args, model, noise_scheduler, dataloader) -> None:
        self.args = args
        self.model = model  # use to predict and remove noises
        self.noise_scheduler = noise_scheduler  # use to add noise to images
        self.train_dataloader = dataloader
        self.val_dataloader = DataLoader(IClevrDataSet(root=args.data_folder_root, mode='val')
                                         , batch_size=args.batch_size, shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.MSELoss()

        self.last_epoch = 0
        if self.args.pretrained_load_path is not None:
            self.loadModel()

        self.inv_normalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],  # -mean / std for each channel
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]  # 1 / std for each channel
        )

    def train(self):
        best_loss = np.inf
        for epoch in range(self.last_epoch, self.args.n_epoch):
            self.model.train()
            epoch_loss = []
            pbar = tqdm(enumerate(self.train_dataloader))
            for i, (img, label) in pbar:
                img = img.to(self.args.device)
                label = label.to(self.args.device)

                # add noise to image
                # use noise scheduler to add random noise to the image
                noise = torch.randn_like(img)
                timestep = torch.randint(0, 999, (img.shape[0], )).long().to(self.args.device)
                noise_img = self.noise_scheduler.add_noise(img, noise, timestep)

                # predict noise
                # predict the noise added by the noise scheduler
                # we want the predicted noise be similar to the noise we randomed 
                pred_noise = self.model(noise_img, label, timestep)

                # calculate distance between the random noise and the predicted noise
                loss = self.criterion(pred_noise, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss.append(loss.item())
                pbar.set_description_str(f"epoch: {epoch + 1} / {self.args.n_epoch}, iter: {i + 1} / {len(self.train_dataloader)}, loss: {np.mean(epoch_loss)}")
            print(f'[Epoch {epoch+1}] loss = {np.mean(epoch_loss)}')

            if np.mean(epoch_loss) < best_loss:
                self.saveModel(epoch)
            # if (epoch+1) % 5 == 0 or epoch == 0:
                self.eval(epoch)

    def eval(self, epoch=0):
        denoising_process_img = []
        if self.args.with_test:
            evaluator = evaluation_model()
        self.model.eval()
        with torch.no_grad():
            avg_acc = []
            for label in tqdm(self.val_dataloader):
                x = torch.randn(label.shape[0], 3, 64, 64).to(self.args.device)
                label = label.to(self.args.device)

                # save denoising process image
                denoising_process_img.append(x[0].unsqueeze(0))

                for t in tqdm(self.noise_scheduler.timesteps):
                    pred_noise = self.model(x, label, t)
                    x = self.noise_scheduler.step(pred_noise, t, x).prev_sample

                    # save denoising process image
                    if (t+1) % self.args.save_denoised_fig_per == 0:
                        denoising_process_img.append(self.inv_normalize(x[0].unsqueeze(0)))

                # calculate accuracy
                if self.args.with_test:
                    acc = evaluator.eval(x, label)
                    print(f'Label: {label}, Acc: {acc:.5f}')
                    avg_acc.append(acc)
                
                # save images
                img_save_path = os.path.join(self.args.image_save_path, f'img_grid_epoch={epoch+1}.png')
                denoise_save_path = os.path.join(self.args.image_save_path, f'denoise_grid_epoch={epoch+1}.png')
                x = self.inv_normalize(x)   # inverse the normalization of images
                pltImageGrid(x, img_save_path)
                pltDenoiseProcess(torch.cat(denoising_process_img, 0), denoise_save_path)
            if self.args.with_test:
                print(f'Avg accuracy: {np.mean(avg_acc)}')
    
    def saveModel(self, epoch):
        print(f'> Saving model to {self.args.ckpt_save_path}...')
        torch.save({
            'epoch': epoch+1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.args.ckpt_save_path)

    def loadModel(self):
        print(f'> Loading model from {self.args.pretrained_load_path}...')
        checkpoint = torch.load(self.args.pretrained_load_path)
        self.last_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        

#############   Utils   #############
def pltDenoiseProcess(images, path):
    plt.imshow(torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(images, nrow=11)))
    plt.savefig(path)

def pltImageGrid(images, path):
    plt.imshow(torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(images, nrow=8)))
    plt.savefig(path)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder_root',   type=str,   default='')
    parser.add_argument('--mode',               type=str,   default='train',    choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size',         type=int,   default=64)
    parser.add_argument('--n_epoch',            type=int,   default=300)
    parser.add_argument('--device',             type=str,   default='cuda:1',     choices=['cuda:1', 'cpu'])
    parser.add_argument('--learning_rate',      type=float, default=1e-4)
    parser.add_argument('--ckpt_save_path',     type=str,   default='results/ckpt/last.pt')
    parser.add_argument('--image_save_path',    type=str,   default='results/img')
    parser.add_argument('--pretrained_load_path',       type=str,   default=None)
    parser.add_argument('--with_test',          action='store_true')
    parser.add_argument('--save_denoised_fig_per',      type=int,   default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = getArgs()

    # prepare data
    dataset = IClevrDataSet(root=args.data_folder_root, mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # prepare model and scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    model = ConditionDDPM(img_channel=3, num_class=24).to(args.device)    # image channel: RGB

    # start training
    trainer = DDPMTrainer(args, model, noise_scheduler, dataloader)
    trainer.train()
    # trainer.eval(0)