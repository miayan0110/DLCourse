import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

def evaluate(net, data, device):
    # implement the evaluation function here
    avg_score = 0
    net.eval()
    with torch.no_grad():
        for sample in tqdm(data):
            img = torch.tensor(sample['image']).float().to(device)
            mask = torch.tensor(sample['mask']).long().to(device)

            pred = net(img)
            avg_score += utils.dice_score(pred, mask)
        print(f'Average dice score = {avg_score/len(data)}')
    
    return avg_score/len(data)
