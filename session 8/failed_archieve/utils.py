import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm 

DEVICE=None
SEED = 2**32-1

def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    
    if torch.cuda.is_available():
        DEVICE='cuda'
    elif torch.backends.mps.is_available():
        DEVICE='mps'
    else:
        DEVICE='cpu'
    print(f'Device selected {DEVICE}')
    return DEVICE

def set_seed(mix_precision:bool=False):
    torch.cuda.amp.autocast(enabled=mix_precision)
    # REPRODUCE
    torch.backends.cudnn.deterministic=True
    random.seed(hash('setting random seeds')% SEED)
    np.random.seed(hash('improves reproducibility')%SEED)
    torch.manual_seed(hash("by removing stochasticity")%SEED)
    torch.cuda.manual_seed_all(hash('so runs are repetable')%SEED)



device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
cuda   = torch.cuda.is_available()

from matplotlib import pyplot as plt
def show_egs(loader,figsize):
    batch_data,batch_label = next(iter(loader))
    _ = plt.figure(figsize=figsize)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        img = batch_data[i].permute(1, 2, 0).cpu().numpy().squeeze()
        plt.imshow(img,cmap='gray')
        label = loader.dataset.classes[ int(batch_label[i].cpu().numpy()) ]
        plt.title(f'label: {label}')
    plt.show()   

