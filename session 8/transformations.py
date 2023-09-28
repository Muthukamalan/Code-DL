from torchvision import datasets, transforms
from dataloader import Cifar10D

_cifar = Cifar10D()


std_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        _cifar.mean,
        _cifar.stds
    )
])

train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
             _cifar.mean,
            _cifar.stds
        )
    ])