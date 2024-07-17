# implement your training script here
import Dataloader as dl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
        if self.args.train_mode == 'FT':
            # load model here
            self.load('LOSO')

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

                epoch_loss += loss.item()
                pred = torch.argmax(pred, dim=1)
                epoch_correct += (pred == label).sum().item()

            print(f'[Epoch {i+1:3d} ] loss = {epoch_loss / len(dataloader):.9f} acc = {epoch_correct*100 / len(dataloader.dataset)}')
            losses.append(epoch_loss / len(dataloader))
            acc.append(epoch_correct*100 / len(dataloader.dataset))

            if acc.index(max(acc)) == i:
                self.save()
        return losses, acc

    def subjectDependent(self):
        return DataLoader(dataset=dl.MIBCI2aDataset(self.args.expri_mode, './dataset/SD')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def leaveOneSubjectOut(self):
        return DataLoader(dataset=dl.MIBCI2aDataset(self.args.expri_mode, './dataset/LOSO')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def withFinetune(self):
        return DataLoader(dataset=dl.MIBCI2aDataset(self.args.expri_mode, './dataset/FT')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def load(self, method):
        path = self.args.save_path + method + '.pth'

        print(f'> Loading model from {path}...')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self):
        path = self.args.save_path + self.args.train_mode + '.pth'

        print(f'> Saving model to {path}...')
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}
                    , path)