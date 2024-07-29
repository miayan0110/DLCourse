import argparse
from tqdm import tqdm
import oxford_pet as opData
from utils import *
from models import unet
from models import resnet34_unet
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='DL_Lab3_UNet_313551073_顏琦恩.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, default='./dataset/', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='batch size')
    parser.add_argument('--model_to_use', '-u', type=str, default='resnet', help='model to inference (unet/resnet)')
    parser.add_argument('--plot', '-p', type=str, default='n', help='plot during inferencing or not (y/n)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    dataloader = DataLoader(opData.load_dataset(args.data_path, 'test'), batch_size=args.batch_size, shuffle=True)
    if args.model_to_use == 'unet':
        model = load_model(f'./saved_models/{args.model}', unet.UNet(3))
    elif args.model_to_use == 'resnet':
        model = load_model(f'./saved_models/{args.model}', resnet34_unet.ResNet34_UNet(3))

    avg_score = []
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader):
            img = torch.tensor(sample['image']).float().to(args.device)
            mask = torch.tensor(sample['mask']).float().to(args.device)

            pred = model(img)
            score = dice_score(pred, mask)
            avg_score.append(score)
            if max(avg_score) == score and args.plot == 'y':
                plot_grid(img, mask, pred)
        print(f'Inference dice score = {sum(avg_score)/len(avg_score)}')