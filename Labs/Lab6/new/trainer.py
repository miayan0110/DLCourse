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

from dataloader import IClevrDataSet
from model import ConditionDDPM
from evaluator import evaluation_model

from torch.utils.tensorboard import SummaryWriter

#############   Trainer   #############
class DDPMTrainer:
    def __init__(self, args, model, noise_scheduler, dataloader, writer) -> None:
        self.args = args
        self.model = model  # use to predict and remove noises
        self.noise_scheduler = noise_scheduler  # use to add noise to images
        self.train_dataloader = dataloader
        self.val_dataloader = DataLoader(IClevrDataSet(root=args.data_folder_root, mode='val')
                                         , batch_size=args.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.MSELoss()
        self.writer = writer

        self.last_epoch = 0
        if self.args.pretrained_load_path is not None:
            self.loadModel()
            # self.evaluator = evaluation_model()

        self.inv_normalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],  # -mean / std for each channel
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]  # 1 / std for each channel
        )

    def train(self):
        for epoch in range(self.last_epoch, self.args.n_epoch):
            epoch_loss = []
            for img, label in tqdm(self.train_dataloader):
                img = img.to(self.args.device)
                label = label.to(self.args.device)

                # add noise to image
                # use noise scheduler to add random noise to the image
                noise = torch.rand_like(img)
                timestep = torch.randint(0, 999, (img.shape[0],)).long().to(self.args.device)
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
            print(f'[Epoch {epoch+1}] loss = {sum(epoch_loss) / len(epoch_loss):.5f}')
            self.writer.add_scalars('Training Loss', {'Loss': sum(epoch_loss) / len(epoch_loss)}, self.args.n_epoch)

            # with torch.no_grad():
            #     denoised_img = self.inv_normalize(self.noise_scheduler.step(pred_noise, timestep, noise).prev_sample)
            #     save_path = os.path.join(self.args.image_save_path, f'epoch={epoch+1}.png')
            #     pltImageGrid(denoised_img, save_path)

            self.saveModel(epoch)
            if (epoch+1) % 5 == 0 or epoch == 0:
                self.eval(epoch)

    def eval(self, epoch=0):
        with torch.no_grad():
            for label_name, label in tqdm(self.val_dataloader):
                x = torch.randn(label.shape[0], 3, 64, 64).to(self.args.device)
                label = label.to(self.args.device)

                for t in tqdm(self.noise_scheduler.timesteps):
                    pred_noise = self.model(x, label, t)
                    x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
                
                x = self.inv_normalize(x)
                save_path = os.path.join(self.args.image_save_path, f'epoch={epoch+1}.png')
                pltImageGrid(x, save_path)
                break
                

    
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
    parser.add_argument('--batch_size',         type=int,   default=8)
    parser.add_argument('--n_epoch',            type=int,   default=100)
    parser.add_argument('--device',             type=str,   default='cuda',     choices=['cuda', 'cpu'])
    parser.add_argument('--learning_rate',      type=float, default=1e-4)
    parser.add_argument('--ckpt_save_path',     type=str,   default='results/ckpt/last.pt')
    parser.add_argument('--image_save_path',    type=str,   default='results/img')
    parser.add_argument('--pretrained_load_path',     type=str,   default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = getArgs()

    writer = SummaryWriter(os.path.join(args.data_folder_root, 'results/tensorboard'))

    # prepare data
    dataset = IClevrDataSet(root=args.data_folder_root, mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # prepare model and scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    model = ConditionDDPM(img_channel=3, num_class=24).to(args.device)    # image channel: RGB

    # start training
    trainer = DDPMTrainer(args, model, noise_scheduler, dataloader, writer)
    trainer.train()
    # trainer.eval(14)