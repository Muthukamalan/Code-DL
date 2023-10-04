from typing import Any, Callable, List, Optional
from torchvision import datasets
import numpy as np
import torch
from torch import nn

class CIFAR10Dataset(nn.Module):
    def __init__(
        self,
        root:str,
        train:bool,
        Atransforms:Optional[Callable],
        download:bool=False
    ):
        '''
            root : path_to_dir,
            train: Training Dataset or Testing Dataset,
            Atransforms: Albumentation Transforms,
            download: downloadable or not
        '''
        self.transforms = Atransforms
        self.ds = datasets.CIFAR10(root='../data/',train=train)
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self,idx:int):
        '''
            Apply Albumentation Transformation of Image when calls
        '''
        image,label = self.ds[idx]
        image = np.array(image)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return (image,label)






class CIFAR10DataLoader:
    def __init__(self,batch_size=128,is_cuda_available=False)->None:
        '''
            batch_size: Size of the batch
            is_cuda_available: True or False
        '''
        self.batch_size: int = batch_size

        self.means: List[float] = [0.4914, 0.4822, 0.4465]
        self.stds: List[float] = [0.2470, 0.2435, 0.2616]

        self.dataloader_args = {"shuffle": True, "batch_size": self.batch_size}
        if is_cuda_available:
            self.dataloader_args["num_workers"] = 8
            self.dataloader_args["pin_memory"] = True
        self.classes: List[str] = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    def get_loader(self,train_dataset,test_dataset):
        ''' dataset: Accepts Train or Test Dataset'''
        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        return (
            torch.utils.data.DataLoader(self.train_dataset,**self.dataloader_args), torch.utils.data.DataLoader(self.test_dataset,**self.dataloader_args)
        )
    
    def classes(self):
        return self.classes