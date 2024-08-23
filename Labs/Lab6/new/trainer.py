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
    def __init__(self, args, model, noise_scheduler) -> None:
        self.args = args
        self.model = model  # use to predict and remove noises
        self.noise_scheduler = noise_scheduler  # use to add noise to images
        self.train_dataloader = DataLoader(IClevrDataSet(root=args.data_folder_root, mode='train')
                                         , batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(IClevrDataSet(root=args.data_folder_root, mode='val', file=args.test_json_file)
                                         , batch_size=1, shuffle=False)
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
                best_loss = np.mean(epoch_loss)
                self.saveModel(epoch)
                self.eval(epoch)

    def eval(self, epoch=0):
        result_imgs = []
        result_labels = []
        denoising_process_img = []
        if self.args.with_test:
            evaluator = evaluation_model()

        self.model.eval()
        with torch.no_grad():
            for i, label in tqdm(enumerate(self.val_dataloader)):
                x = torch.randn(label.shape[0], 3, 64, 64).to(self.args.device)
                label = label.to(self.args.device)
                result_labels.append(label)

                for t in tqdm(self.noise_scheduler.timesteps):
                    pred_noise = self.model(x, label, t)
                    x = self.noise_scheduler.step(pred_noise, t, x).prev_sample

                    # save denoising process image
                    if ((t+1) % self.args.save_denoised_fig_per == 0 or t == 0) and i == 25:
                            denoising_process_img.append(self.inv_normalize(x))
                result_imgs.append(x)

            # calculate accuracy
            imgs = torch.cat(result_imgs, 0)
            labels = torch.cat(result_labels, 0)
            if self.args.with_test:
                acc = evaluator.eval(imgs, labels)
                print(f'Avg accuracy: {acc}')
            
            # save images
            img_save_path = os.path.join(self.args.image_save_path, f'img_grid_epoch={epoch+1}_{self.args.test_json_file}.png')
            denoise_save_path = os.path.join(self.args.image_save_path, f'denoise_grid_epoch={epoch+1}_{self.args.test_json_file}.png')
            imgs = self.inv_normalize(imgs)   # inverse the normalization of images
            pltImageGrid(imgs, img_save_path)
            if len(denoising_process_img) > 1:
                pltDenoiseProcess(torch.cat(denoising_process_img, 0), denoise_save_path)
                
    
    def saveModel(self, epoch):
        print(f'> Saving model to {self.args.ckpt_save_path}...')
        torch.save({
            'epoch': epoch+1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.args.ckpt_save_path)

    def loadModel(self):
        print(f'> Loading model from {self.args.pretrained_load_path}...')
        checkpoint = torch.load(self.args.pretrained_load_path, map_location=self.args.device)
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
    parser.add_argument('--mode',               type=str,   default='val',    choices=['train', 'val'])
    parser.add_argument('--batch_size',         type=int,   default=64)
    parser.add_argument('--n_epoch',            type=int,   default=300)
    parser.add_argument('--device',             type=str,   default='cuda',     choices=['cuda', 'cuda:1', 'cpu'])
    parser.add_argument('--learning_rate',      type=float, default=1e-4)
    parser.add_argument('--ckpt_save_path',     type=str,   default='results/ckpt/last.pt')
    parser.add_argument('--image_save_path',    type=str,   default='results/img')
    parser.add_argument('--pretrained_load_path',       type=str,   default=None)
    parser.add_argument('--with_test',          action='store_true')
    parser.add_argument('--save_denoised_fig_per',      type=int,   default=100)
    parser.add_argument('--test_json_file',     type=str,   default='test')

    return parser.parse_args()


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    args = getArgs()

    # prepare model and scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    model = ConditionDDPM(img_channel=3, num_class=24).to(args.device)    # image channel: RGB

    # start training
    trainer = DDPMTrainer(args, model, noise_scheduler)
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.eval(-1)