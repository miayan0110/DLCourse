import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        files = os.listdir(filePath)
        features = np.load(filePath+files[0])   # (288, 22, 438)
        for i in range(1, len(files)):
            features = np.concatenate((features, np.load(filePath+files[i])), axis=0)
        features = np.expand_dims(features, axis=1)
        return features # (2304, 1, 22, 438)

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        files = os.listdir(filePath)
        labels = np.load(filePath+files[0]) # (288,)
        for i in range(1, len(files)):
            labels = np.concatenate((labels, np.load(filePath+files[i])), axis=0)
        labels = np.expand_dims(labels, axis=1)
        return labels   # (2304, 1)

    def __init__(self, mode, path):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if mode == 'train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath=f'{path}_train/features/')
            self.labels = self._getLabels(filePath=f'{path}_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath=f'{path}/features/')
            self.labels = self._getLabels(filePath=f'{path}/labels/')
        if mode == 'test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath=f'{path}_test/features/')
            self.labels = self._getLabels(filePath=f'{path}_test/labels/')

    def __len__(self):
        # implement the len method
        return len(self.labels)

    def __getitem__(self, idx):
        # implement the getitem method
        return torch.Tensor(self.features[idx]).float(), torch.Tensor(self.labels[idx]).long()