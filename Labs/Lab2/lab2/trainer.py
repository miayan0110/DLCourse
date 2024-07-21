# implement your training script here
import Dataloader as dl
from tester import Tester
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
                self.training_set = self.subjectDependent
            case 'LOSO':
                self.training_set = self.leaveOneSubjectOut
            case 'FT':
                self.training_set = self.withFinetune

    def train(self):
        losses = []
        acc = []
        test_acc = 0.0
        best_test_acc_at_epoch = 0
        # load model here when fine tuning
        if self.args.train_mode == 'FT':
            self.load('LOSO')

        # start training
        for i in range(self.args.epoch):
            cost = 0.0
            correct = 0.0
            dataloader = self.training_set()
            self.model.train()

            for feature, label in tqdm(dataloader):
                feature = feature.to(self.args.device)
                label = label.squeeze(1).to(self.args.device)
                self.optimizer.zero_grad()

                pred = self.model(feature)
                loss = self.loss_function(pred, label)
                loss.backward()

                self.optimizer.step()

                # calculating epoch loss and accuracy 
                cost += loss.item()
                pred = torch.argmax(pred, dim=1)
                correct += (pred == label).sum().item()

            epoch_loss = cost / len(dataloader)
            epoch_acc = correct*100 / len(dataloader.dataset)
            print(f'[Epoch {i+1:3d} ] loss = {epoch_loss:.9f} acc = {epoch_acc}')
            losses.append(epoch_loss)
            acc.append(epoch_acc)

            # save model
            new_test_acc = self.getTestAccuracy()
            if new_test_acc > test_acc:
                test_acc = new_test_acc
                best_test_acc_at_epoch = i
                self.save()
        return losses, acc, best_test_acc_at_epoch

    def getTestAccuracy(self):
        tester = Tester(self.args, model=self.model)
        return tester.test()

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