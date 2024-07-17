# script for drawing figures, and more if needed
from model import SCCNet
from trainer import Trainer
from tester import Tester
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(args, loss, acc):
    fig, ax = plt.subplots(1, 2)
    plt.suptitle(f'Loss and Accuracy Curve (Training Method: {args.train_mode})')

    ax[0].plot(np.arange(len(loss)), loss)
    ax[0].set(xlabel='Epoch', ylabel='Loss', title='Learning Curve')

    ax[1].plot(np.arange(len(acc)), acc)
    ax[1].set(xlabel='Epoch', ylabel='Accuracy(%)', title='Accuracy Curve')
    
    plt.show()

def main(args):
    if args.expri_mode == 'finetune' and args.train_mode != 'FT':
        print('FT dataset should be used when experiment mode is finetune.')
        return
    elif args.expri_mode == 'train' and args.train_mode == 'FT':
        print('FT dataset should not be used when training or testing.')
        return
    
    model = SCCNet.SCCNet(numClasses=4, timeSample=438, Nu=22, C=1, Nc=20, Nt=1, dropoutRate=0.5).to(args.device)

    if args.expri_mode != 'test':
        print('Start training...')
        trainer = Trainer(args, model=model)
        train_loss, train_acc = trainer.train()
        plot(args, train_loss, train_acc)

        args.expri_mode = 'test'

    print('\nStart testing...')
    tester = Tester(args, model=model)
    tester.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--epoch',      type=int,   default=1000)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--expri_mode', type=str,   default='train')    # train/finetune/test
    parser.add_argument('--train_mode', type=str,   default='LOSO')       # SD/LOSO/FT
    parser.add_argument('--save_path',  type=str,   default='./model/')

    args = parser.parse_args()
    main(args)