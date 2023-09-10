from torchvision import datasets, transforms

train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


custom_augmentation = A.Compose([
    # A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(rotate_limit=7,p=0.5),
    A.CoarseDropout(max_holes = 1, max_height=10, max_width=10, min_holes = 1, min_height=8,min_width=8, fill_value=(0.13065974414348602),p=0.4),
    A.geometric.transforms.ElasticTransform(alpha=0,sigma=1,p=1,alpha_affine=2,interpolation=1),
    A.Normalize((0.1307,), (0.3081,)),
    ToTensorV2(),
])

class MNISTDataset(datasets.MNIST):
    def __init__(self,root='../data',train=True,download=True,transform=None):
        super().__init__(root=root,train=train,download=download,transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        image,label = self.data[idx],self.targets[idx]

        if self.transform is not None:
            image = self.transform(image=np.array(image.squeeze()))['image']
        return image, label