import torch
from torchvision import datasets


def get_loaders(
    train_transforms,
    test_transforms,
    num_workers,
    batch_size=32,
    pin_memory=True,
    download=False
):
    train_data = datasets.CIFAR10(
        root='../../data/',
        download=download,
        transform=train_transforms,
        train=True
    )
    test_data = datasets.CIFAR10(
        root='../../data/',
        download=download,
        transform=test_transforms,
        train=False
    )
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)
    return train_loader,test_loader