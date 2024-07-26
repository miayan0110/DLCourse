import torch
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    smooth = 1e-6
    score = ((2*(pred_mask*gt_mask).sum()+smooth)/(pred_mask.sum()+gt_mask.sum()+smooth)).float().mean()
    return score

def load_model(path, model):
    print(f'> Loading model from {path}...')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    return model

def save_model(path, model, optimizer, epoch):
    print(f'> Saving model to {path}...')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_epoch': epoch
        }, path)

def plot_pred(pred_mask, image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    image = image.squeeze().permute(1, 2, 0)
    axes[0].imshow(image.cpu().detach().numpy()/255)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    pred_mask = pred_mask.squeeze()
    axes[1].imshow(pred_mask.cpu().detach().numpy(), cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    plt.suptitle("Comparison of Predicted Mask and Original Image", fontsize=16)
    
    plt.show()

def plot_loss(loss, model_used):
    plt.plot(loss, linestyle='-', color='b')
    plt.title(f'Loss of {model_used}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()