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