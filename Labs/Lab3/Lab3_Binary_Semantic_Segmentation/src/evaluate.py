import torch
from tqdm import tqdm
from utils import *

def evaluate(net, data, device):
    # implement the evaluation function here
    avg_score = []
    net.eval()
    with torch.no_grad():
        for sample in tqdm(data):
            img = torch.tensor(sample['image']).float().to(device)
            mask = torch.tensor(sample['mask']).float().to(device)

            pred = net(img)
            avg_score.append(dice_score(pred, mask))
    
    return sum(avg_score)/len(avg_score)
