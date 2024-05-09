import os
import time
from torch.optim import Adam
import numpy as np
from torchviz import make_dot
from PIL import Image
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustDataset(torch.utils.data.Dataset):
    """Customized pytorch Dataset class inheriting from torch.utils.data.Dataset. 
    The class allows to handle data for autoencoders as well with "add_noise" and "no_labels" options"""
    def __init__(self, subset, transform=transforms.ToTensor(), add_noise=False, no_labels=False):
        self.subset = subset
        self.transform = transform
        self.add_noise = add_noise
        self.no_labels = no_labels
    def __getitem__(self, indices):
        x, y = self.subset[indices]
        if(self.add_noise):
            x = self.transform(x)
            x = x+(0.8)*torch.randn(x.shape)
            return torch.clamp(x, min=0, max=1) 
        elif(self.no_labels):
            return self.transform(x)
        else:
            x = self.transform(x)
            return x, y 
    def __len__(self):
        return len(self.subset)