import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader as imgloader


#############   Dataset   #############
def getData(mode):
    label_dict = json.load(open('./objects.json'))
    if mode == 'train':
        data_dict = json.load(open('./train.json'))
        
        img_path = []
        labels = []
        for key, value in data_dict.items():
            label = [1 if x in value else 0 for x in label_dict.keys()] # to one-hot vector
            img_path.append(key)
            labels.append(label)
        return img_path, labels
    else:
        if mode == 'val':
            data_list = json.load(open('./test.json'))
        else:
            data_list = json.load(open('./new_test.json'))

        labels = []
        for value in data_list:
            label = [1 if x in value else 0 for x in label_dict.keys()] # to one-hot vector
            labels.append(label)
        return labels


class IClevrDataSet(torch.utils.data.Dataset):
    def __init__(self, root, mode='train') -> None:
        super().__init__()
        self.root = root
        self.mode = mode
        if self.mode == 'train':
            self.imgs, self.labels = getData(mode)
        else:
            self.labels = getData(mode)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # data preprocessing
        if self.mode == 'train':
            transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = imgloader(os.path.join(self.root, 'iclevr', self.imgs[index]))
            img = transformer(img)

            return img, torch.Tensor(self.labels[index])
        else:
            return torch.Tensor(self.labels[index])
    



# if __name__ == '__main__':
#     dataset = IClevrDataSet(root='', mode='train')
#     data2plt = DataLoader(dataset, batch_size=8, shuffle=False)
#     img, label = next(iter(data2plt))
#     pltImageGrid(img)