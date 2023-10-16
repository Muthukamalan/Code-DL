import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


train_transforms = A.Compose([
    A.HorizontalFlip(always_apply=False,p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, mask_value=(0.4914, 0.4822, 0.4465), always_apply=False,p=0.5,),
    A.CoarseDropout( max_holes= 1,max_height = 16,max_width = 16,min_holes= 1,min_height = 16,min_width= 16,mask_fill_value = (0.4914, 0.4822, 0.4465),always_apply = False, p= 0.5),
    A.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2470, 0.2435, 0.2616),p=1),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2470, 0.2435, 0.2616),p=1),
    ToTensorV2(),
])