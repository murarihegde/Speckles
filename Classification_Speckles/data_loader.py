import numpy as np
import torch
from torchvision import datasets
import os 
from torch.utils.data import DataLoader, Dataset
from torchvision import T

## The Folder in which the data is stored

class speckle_dataset(Dataset):
    def __init__(self, batch_size, root_dir, transform=None):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform 

    def get_data():
        train_set = datasets.ImageFolder(root_dir + '/train', transform=transform)
        test_set = datasets.ImageFolder(root_dir + '/test', transform=transform)
        train = DataLoader(train_set, batch_size=4, shuffle=True)
        test = DataLoader(test_set, batch_size=4, shuffle=True)

        return train, test
            
