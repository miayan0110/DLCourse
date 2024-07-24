import numpy as np
import torch

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred_mask = pred_mask.cpu().detach().numpy()
    gt_mask = gt_mask.cpu().detach().numpy()
    score = 2 * np.sum(pred_mask * gt_mask) / (np.sum(pred_mask) + np.sum(gt_mask))
    return score

def load_model(path, model):
    print(f'> Loading model from {path}...')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    return model

def save_model(path, model):
    print(f'> Saving model to {path}...')
    torch.save(model.state_dict(), path)