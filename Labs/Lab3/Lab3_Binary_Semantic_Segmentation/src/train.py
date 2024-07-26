import argparse
import oxford_pet as opData
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import unet
from models import resnet34_unet
from evaluate import *
from utils import *

def train(args):
    # implement the training function here
    train_dataloader = DataLoader(opData.load_dataset(args.data_path, 'train'), batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(opData.load_dataset(args.data_path, 'valid'), batch_size=args.batch_size, shuffle=True)

    if args.model == 'unet':
        model = unet.UNet(3).to(args.device)
    elif args.model == 'resnet':
        model = resnet34_unet.ResNet34_UNet(3).to(args.device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # start training
    losses = []
    scores = []
    max_score = 0
    model.train()
    for epoch in range(args.epochs):
      
        epoch_cost = []
        epoch_score = []
        for sample in tqdm(train_dataloader):
            img = torch.Tensor(sample['image']).float().to(args.device)
            mask = torch.Tensor(sample['mask']).float().to(args.device)
            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            epoch_cost.append(loss)
            epoch_score.append(dice_score(pred, mask))
        
        losses.append((sum(epoch_cost)/len(epoch_cost)).item())
        scores.append((sum(epoch_score)/len(epoch_score)).item())
        print(f'[Epoch {epoch+1}] loss = {losses[epoch]:.9f}, average dice score = {scores[epoch]:.9f}')

        # validation
        valid_score = evaluate(model, valid_dataloader, args.device)
        print(f'validation dice score = {valid_score}')
        if valid_score > max_score:
            save_model(f'{args.save_path + args.model}.pth', model, optimizer, epoch)
            max_score = valid_score
    plot_loss(losses, args.model)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path',              type=str,   default='./dataset/', help='path of the input data')
    parser.add_argument('--epochs', '-e',           type=int,   default=2, help='number of epochs')
    parser.add_argument('--batch_size', '-b',       type=int,   default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr',   type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model','-m',             type=str,   default='resnet', help='model to train (unet/resnet)')
    parser.add_argument('--device','-d',            type=str,   default='cuda', help='the device models train on')
    parser.add_argument('--save_path','-p',         type=str,   default='./saved_models/', help='the device models train on')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    # opData.SimpleOxfordPetDataset.download(args.data_path)

    train(args)