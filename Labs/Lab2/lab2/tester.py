# implement your testing script here
import Dataloader as dl
from model import SCCNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Tester:
    def __init__(self, args, model):
        self.model = model
        self.args = args
        self.loss_function = nn.CrossEntropyLoss()

        assert self.args.train_mode in ['SD', 'LOSO', 'FT']
        match self.args.train_mode:
            case 'SD':
                self.training_set = self.subjectDependent
            case 'LOSO'|'FT':
                self.training_set = self.leaveOneSubjectOut

    def test(self):
        losses = 0.0
        correct = 0

        # if test is called during training, do not load new model, use current model to do the test
        if not self.args.use_current_model:
            self.load()
        self.model.eval()

        dataloader = self.training_set()
        with torch.no_grad():
            for feature, label in tqdm(dataloader):
                feature = feature.to(self.args.device)
                label = label.squeeze(1).to(self.args.device)

                pred = self.model(feature)
                loss = self.loss_function(pred, label)

                losses += loss.item()
                pred = pred.argmax(dim=1)
                correct += (pred == label).sum().item()
            
            print(f'loss = {losses / len(dataloader):.9f} acc = {correct*100 / len(dataloader.dataset)}')
            return correct*100 / len(dataloader.dataset)

    def load(self):
        path = self.args.save_path + self.args.train_mode + self.args.test_model + '.pth'
        
        print(f'> Loading model from {path}...')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])

    def subjectDependent(self):
        return DataLoader(dataset=dl.MIBCI2aDataset('test', './dataset/SD')
                          , batch_size=self.args.batch_size
                          , shuffle=True)

    def leaveOneSubjectOut(self):
        return DataLoader(dataset=dl.MIBCI2aDataset('test', './dataset/LOSO')
                          , batch_size=self.args.batch_size
                          , shuffle=True)