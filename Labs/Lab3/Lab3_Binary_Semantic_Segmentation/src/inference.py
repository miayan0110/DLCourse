import argparse
from tqdm import tqdm
import oxford_pet as opData
from utils import *
from models import unet
from models import resnet34_unet

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default='./dataset/', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    data = opData.load_dataset(args.data_path, 'test')
    model = load_model(f'./saved_models/{args.model}', unet.UNet(64, 9))
    # model = load_model(f'./saved_models/{args.model}', resnet34_unet.UNet_ResNet34(64, 7))

    unet.eval()
    with torch.no_grad():
        for sample in tqdm(data):
            img = torch.tensor(sample['image']).float().to(args.device)
            mask = torch.tensor(sample['mask']).long().to(args.device)

            pred = model(img)