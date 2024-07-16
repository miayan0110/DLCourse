# implement your training script here
import Dataloader as dl
from model import SCCNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

class Trainer:
    def __init__(self, args, model):
        self.model = model
        self.args = args
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0001)    # weight_decay: l2 regularzation coefficient

        assert self.args.train_mode in ['SD', 'LOSO', 'FT']
        match self.args.train_mode:
            case 'SD':
                self.training_method = self.subjectDependent
            case 'LOSO':
                self.training_method = self.leaveOneSubjectOut
            case 'FT':
                self.training_method = self.withFinetune

    def train(self):
        losses = []
        acc = []

        for i in range(self.args.epoch):
            epoch_loss = 0.0
            epoch_correct = 0.0
            dataloader = self.training_method()
            self.model.train()

            for feature, label in tqdm(dataloader):
                feature = feature.to(self.args.device)
                label = label.squeeze(1).to(self.args.device)
                self.optimizer.zero_grad()

                pred = self.model(feature)
                loss = self.loss_function(pred, label)
                loss.backward()

                self.optimizer.step()

                epoch_loss += loss
                pred = pred.argmax(dim=1)
                epoch_correct += (pred == label).sum()
            print(f'[Epoch {i+1:3d} ] loss = {epoch_loss / len(dataloader):.9f} acc = {epoch_correct / len(dataloader.dataset)}')
            losses.append(epoch_loss / len(dataloader))
            acc.append(epoch_correct / len(dataloader.dataset))


    def subjectDependent(self):
        return DataLoader(dataset=dl.MIBCI2aDataset(self.args.expri_mode, './dataset/SD')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def leaveOneSubjectOut(self):
        return DataLoader(dataset=dl.MIBCI2aDataset(self.args.expri_mode, './dataset/LOSO')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def withFinetune(self):
        # load model here
        return DataLoader(dataset=dl.MIBCI2aDataset(self.args.expri_mode, './dataset/FT')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def save(self):
        path = self.save_path + self.train_mode + '.pth'
        torch.save(self.model.state_dict(), path)
        print(f'> Save model to {path}...')


def main(args):
    if args.expri_mode == 'finetune' and args.train_mode != 'FT':
        print('FT dataset should be used when experiment mode is finetune.')
        return
    elif args.expri_mode in ('train', 'test') and args.train_mode == 'FT':
        print('FT dataset should not be used when training or testing.')
        return
    
    model = SCCNet.SCCNet(numClasses=4, timeSample=288, Nu=22, C=1, Nc=22, Nt=1, dropoutRate=0.5).to(args.device)

    trainer = Trainer(args, model=model)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--epoch',      type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=0.5)
    parser.add_argument('--expri_mode', type=str,   default='train')    # train/finetune/test
    parser.add_argument('--train_mode', type=str,   default='SD')       # SD/LOSO/FT
    parser.add_argument('--save_path',  type=str,   default='./model/')

    args = parser.parse_args()
    main(args)