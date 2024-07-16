# implement your training script here
import Dataloader as dl
from model import SCCNet
import torch
from torch.utils.data import DataLoader
import argparse

class Trainer:
    def __init__(self, args, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.epoch = args.epoch
        self.save_path = args.save_path

        assert args.train_mode in ['SD', 'LOSO', 'FT']
        self.train_mode = args.train_mode

        match self.train_mode:
            case 'SD':
                self.training_method = self.subjectDependent
            case 'LOSO':
                self.training_method = self.leaveOneSubjectOut
            case 'FT':
                self.training_method = self.withFinetune

    def train(self):
        losses = []

        for i in range(self.epoch):
            pass

    def validation(self):
        pass

    def subjectDependent(self):
        pass

    def leaveOneSubjectOut(self):
        pass

    def withFinetune(self):
        pass

    def save(self):
        path = self.save_path + self.train_mode + '.pth'
        torch.save(self.model.state_dict(), path)
        print(f'> Save model to {path}...')


def main(args):
    dataloader = DataLoader(dataset=dl.MIBCI2aDataset(args.expri_mode), batch_size=args.batch_size, shuffle=True)
    model = SCCNet(numClasses=4, timeSample=288, Nu=22, C=1, Nc=22, Nt=1, dropoutRate=0.5).to(args.device)

    trainer = Trainer(args, model=model, dataloader=dataloader)
    trainer.train()


if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device: {device}")

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--save_path',  type=str,   default='./model/')
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--epoch',      type=int,   default=1)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--expri_mode', type=str,   default='finetune')
    parser.add_argument('--train_mode', type=str,   default='SD')   # SD/LOSO/FT/ALL

    args = parser.parse_args()
    main(args)


# if __name__ == '__main__':
#     dataset = dl.MIBCI2aDataset('train')
#     f, l = dataset[0:2]
#     model = SCCNet.SCCNet(numClasses=4, timeSample=288, Nu=22, C=1, Nc=22, Nt=1, dropoutRate=0.5)
#     # x = model(torch.unsqueeze(torch.from_numpy(f).float(), dim=0))
#     x = model(torch.from_numpy(f).float())
#     print(x, l)