import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, data_loader):
        self.model.train()
        epoch_loss = []
        for i, data in enumerate(tqdm(data_loader)):
            logits, z_indices = self.model(data.to(self.args.device))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            epoch_loss.append(loss.item())
            loss.backward()
            if i % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

        torch.save(self.model.transformer.state_dict(), self.args.last_checkpoint_path)
        return sum(epoch_loss) / len(epoch_loss)

    @torch.no_grad()
    def eval_one_epoch(self, data_loader):
        self.model.eval()
        epoch_loss = []
        for i, data in enumerate(tqdm(data_loader)):
            logits, z_indices = self.model(data.to(self.args.device))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            epoch_loss.append(loss.item())
        return sum(epoch_loss) / len(epoch_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate, betas=[0.9, 0.96])
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--last_checkpoint_path', type=str, default='./transformer_checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--checkpoint_root', type=str, default='./transformer_checkpoints/', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum_grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save_per_epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start_from_epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt_interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    writer = SummaryWriter('./results/tensorboard')
    
#TODO2 step1-5:    
    min_loss = 10000
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        val_loss = train_transformer.eval_one_epoch(val_loader)

        print(f'[Epoch {epoch}] training loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')
        if train_loss <= min_loss:
            min_loss = train_loss
            torch.save(train_transformer.model.transformer.state_dict(), f'{args.checkpoint_root}min_loss_epoch={epoch}.pt')
            print(f'Saving best checkpoint to {args.checkpoint_root}min_loss_epoch={epoch}.pt...')
        writer.add_scalar('Training loss', train_loss, epoch)
        writer.add_scalar('Evaluating loss', val_loss, epoch)