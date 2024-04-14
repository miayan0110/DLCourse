import pandas as pd
from PIL import Image
from torch.utils import data
import numpy as np

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('training.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('validation.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    else:
        df = pd.read_csv('testing.csv')
        path = df['Path'].tolist()
        return path

class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
        """
        img_p = Image.open(self.root+self.img_name[index])

        # weight, height = img_p.size
        # channel_mode = img_p.mode
        # print("img[{}]: size={}x{}, mode={}".format(index, weight, height, channel_mode))

        """
           step2. Get the ground truth label from self.label
        """
        label = self.label[index]

        """                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
        """
        img_r, img_g, img_b = img_p.split()
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)

        img = []
        img.append(img_r/img_r.max())
        img.append(img_g/img_g.max())
        img.append(img_b/img_b.max())
        img = np.array(img)

        """
            step4. Return processed image and label
        """
        return img, label