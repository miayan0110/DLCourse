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
import gc

def train(args):
    # implement the training function here
    train_dataloader = DataLoader(opData.load_dataset(args.data_path, 'train'), batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(opData.load_dataset(args.data_path, 'valid'), batch_size=args.batch_size, shuffle=True)

    if args.model == 'unet':
        model = unet.UNet(3).to(args.device)
    elif args.model == 'resnet':
        model = resnet34_unet.ResNet34_UNet(3).to(args.device)

    criterion = nn.BCELoss()    # 用於只有binary class的cross entropy
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    losses = []
    scores = []
    max_score = 0

    # 如果要繼續train同一個model，可以使用這個部分，loss會寫入txt檔中，所以可以畫完整的loss curve
    if args.retrain_model != '':
        model, optimizer, max_score  = load_model(f'{args.save_path+args.retrain_model}', model, optimizer)
        losses = load_loss(args.model)

    # start training
    last_epoch = len(losses)    # 從上次訓練到的epoch繼續畫loss curve，如果是train from scratch，就從epoch 0開始畫loss curve
    
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
        print(f'[Epoch {epoch+1}] loss = {losses[-1]:.9f}, average dice score = {scores[-1]:.9f}')

        # validation
        valid_score = evaluate(model, valid_dataloader, args.device)
        print(f'Validation dice score = {valid_score}')
        if valid_score > max_score:
            save_model(f'{args.save_path + args.model}.pth', model, optimizer, epoch, valid_score)
            max_score = valid_score
        save_loss(args.model, losses[last_epoch+epoch:])    # 將loss寫入txt檔
        # 避免gpu out of memory
        gc.collect()
        torch.cuda.empty_cache()

    plot_loss(losses, args.model)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path',              type=str,   default='./dataset/', help='path of the input data')
    parser.add_argument('--epochs', '-e',           type=int,   default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b',       type=int,   default=2, help='batch size')
    parser.add_argument('--learning_rate', '-lr',   type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model','-m',             type=str,   default='resnet', help='model to train (unet/resnet)')
    parser.add_argument('--device','-d',            type=str,   default='cuda', help='the device models train on')
    parser.add_argument('--save_path','-p',         type=str,   default='./saved_models/', help='path where model checkpoints save')
    parser.add_argument('--retrain_model','-r',     type=str,   default='', help='model to retrain')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    # opData.SimpleOxfordPetDataset.download(args.data_path)    # uncomment this line to download the dataset

    train(args)