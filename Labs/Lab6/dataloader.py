import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset as torchData
import numpy as np
from PIL import Image


def custom_transformer():
    transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    return transformer   

def to_one_hot_vector(label_dict, obj_array):
    obj_label = [label_dict.get_label(obj) for obj in obj_array]
    one_hot_vector = [1 if x in obj_label else 0 for x in range(label_dict.get_len())]
    return one_hot_vector

def getData(root, mode):
    label_dict = Labels(root)

    if mode == "train":
        files = []
        labels = []
        with open(os.path.join(root, "train.json"), 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            files.append(key)
            labels.append(to_one_hot_vector(label_dict, value))
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype("long"))
        return files, labels
    
    elif mode == "valid":
        with open(os.path.join(root, "test.json"), 'r') as f:
            data = json.load(f)

        labels = [to_one_hot_vector(label_dict, label) for label in data]
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype("long"))
        return labels
    
    elif mode == "test":
        with open(os.path.join(root, "new_test.json"), 'r') as f:
            data = json.load(f)

        labels = [to_one_hot_vector(label_dict, label) for label in data]
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype("long"))
        return labels

class Labels:
    def __init__(self, root):
        self.path = os.path.join(root, "objects.json")

        with open(self.path, 'r') as f:
            self.dict = json.load(f)

    def get_len(self):
        return len(self.dict)

    def get_label(self, obj):
        return self.dict[obj]
    

class iClevrLoader(torchData):
    def __init__(self, root, mode):
        super().__init__()
        self.root = root
        self.mode = mode

        self.transformer = custom_transformer()
        if mode == "train":
            self.images, self.labels = getData(root, mode)
        else:
            self.labels = getData(root, mode)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if self.mode == "train":
            path = os.path.join(self.root, "iclevr", self.images[index])
            img = self.transformer(Image.open(path).convert("RGB"))
            label = self.labels[index]
            return img, label
        else:
            return self.labels[index]