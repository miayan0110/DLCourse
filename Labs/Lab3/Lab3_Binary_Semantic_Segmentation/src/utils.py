import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    smooth = 1e-6
    score = ((2*(pred_mask*gt_mask).sum()+smooth)/(pred_mask.sum()+gt_mask.sum()+smooth)).float().mean()
    return score

def load_model(path, model, optimizer=None):
    print(f'> Loading model from {path}...')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        max_score = checkpoint['best_valid_score']
        return model, optimizer, max_score
    else:
        return model

def save_model(path, model, optimizer, epoch, valid_score):
    print(f'> Saving model to {path}...')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_epoch': epoch,
        'best_valid_score': valid_score
        }, path)
    
# 用於紀錄這次training的所有loss，以便之後繼續train同個model時可以畫出完整的loss curve
def save_loss(model_name, losses):
    filename = f'{model_name}_loss.txt'
    with open(filename, 'a') as file:
        for loss in losses:
            file.write(f"{loss}\n")

def load_loss(model_name):
    filename = f'{model_name}_loss.txt'
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

# 畫單張圖和他的predicted mask
def plot_pred(image, pred_mask):
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

# 畫多張圖、ground truth、predicted mask，一行最多8張圖
def plot_grid(image, mask, pred_mask):
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle('Comparison of Original Image, Mask, and Predicted Mask', fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(T.ToPILImage()(torchvision.utils.make_grid(image / 255, nrow=8)))
    plt.axis('off')
    plt.title("Original Image")

    fig.add_subplot(3, 1, 2)
    plt.imshow(T.ToPILImage()(torchvision.utils.make_grid(mask, nrow=8)))
    plt.axis('off')
    plt.title("Ground Truth Masks")

    fig.add_subplot(3, 1, 3)
    plt.imshow(T.ToPILImage()(torchvision.utils.make_grid(pred_mask, nrow=8)))
    plt.axis('off')
    plt.title("Predicted Masks")

    plt.show()

# 畫loss
def plot_loss(loss, model_used):
    plt.plot(loss, linestyle='-', color='b')
    plt.title(f'Loss of {model_used}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()