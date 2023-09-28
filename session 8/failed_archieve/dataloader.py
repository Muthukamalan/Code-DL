import os
import torch
from torchvision import datasets

from utils import cuda


class Cifar10D:
    def __init__(self,batch_size=256,is_cuda_available=False)->None:
        self.batch_size = batch_size
        self.mean = [0.4914, 0.4822, 0.4465]
        self.stds = [0.2470, 0.2435, 0.2616]

        self.kwargs = {'shuffle':True,'batch_size':self.batch_size}
        if cuda:
            self.kwargs['num_workers'] = os.cpu_count()-2
            self.kwargs['pin_memory']  = True

    def get_data(self,root,transform,train=True,download=False):
        ds = datasets.CIFAR10(root=root,train=train,download=download,transform=transform)
        self.classes = ds.classes
        self.cls2idx = ds.class_to_idx
        return ds

    def get_loader(self,root,download,transform,train=True):
        loader =  torch.utils.data.DataLoader(
            self.get_data(root=root,transform=transform,train=train,download=download),
            **self.kwargs
        )
        self.idx2cls = [{v:k} for k,v in self.cls2idx.items() ]
        return loader
 