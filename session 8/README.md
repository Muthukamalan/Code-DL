# Insights

## BN
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
         Dropout2d-4           [-1, 16, 32, 32]               0
         ConvLayer-5           [-1, 16, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]           4,608
       BatchNorm2d-7           [-1, 32, 32, 32]              64
              ReLU-8           [-1, 32, 32, 32]               0
         Dropout2d-9           [-1, 32, 32, 32]               0
        ConvLayer-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 16, 32, 32]             512
        MaxPool2d-12           [-1, 16, 16, 16]               0
       TransBlock-13           [-1, 16, 16, 16]               0
           Conv2d-14           [-1, 16, 16, 16]           2,304
      BatchNorm2d-15           [-1, 16, 16, 16]              32
             ReLU-16           [-1, 16, 16, 16]               0
        Dropout2d-17           [-1, 16, 16, 16]               0
        ConvLayer-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,304
      BatchNorm2d-20           [-1, 16, 16, 16]              32
             ReLU-21           [-1, 16, 16, 16]               0
        Dropout2d-22           [-1, 16, 16, 16]               0
        ConvLayer-23           [-1, 16, 16, 16]               0
           Conv2d-24           [-1, 32, 16, 16]           4,608
      BatchNorm2d-25           [-1, 32, 16, 16]              64
             ReLU-26           [-1, 32, 16, 16]               0
        Dropout2d-27           [-1, 32, 16, 16]               0
        ConvLayer-28           [-1, 32, 16, 16]               0
           Conv2d-29           [-1, 16, 16, 16]             512
        MaxPool2d-30             [-1, 16, 8, 8]               0
       TransBlock-31             [-1, 16, 8, 8]               0
           Conv2d-32             [-1, 16, 8, 8]           2,304
      BatchNorm2d-33             [-1, 16, 8, 8]              32
             ReLU-34             [-1, 16, 8, 8]               0
        Dropout2d-35             [-1, 16, 8, 8]               0
        ConvLayer-36             [-1, 16, 8, 8]               0
           Conv2d-37             [-1, 32, 8, 8]           4,608
      BatchNorm2d-38             [-1, 32, 8, 8]              64
             ReLU-39             [-1, 32, 8, 8]               0
        Dropout2d-40             [-1, 32, 8, 8]               0
        ConvLayer-41             [-1, 32, 8, 8]               0
           Conv2d-42             [-1, 64, 8, 8]          18,432
      BatchNorm2d-43             [-1, 64, 8, 8]             128
             ReLU-44             [-1, 64, 8, 8]               0
        Dropout2d-45             [-1, 64, 8, 8]               0
        ConvLayer-46             [-1, 64, 8, 8]               0
           Conv2d-47             [-1, 16, 8, 8]           1,024
        MaxPool2d-48             [-1, 16, 4, 4]               0
       TransBlock-49             [-1, 16, 4, 4]               0
AdaptiveAvgPool2d-50             [-1, 16, 1, 1]               0
           Conv2d-51             [-1, 10, 1, 1]             160
================================================================
Total params: 42,256
Trainable params: 42,256
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.02
Params size (MB): 0.16
Estimated Total Size (MB): 3.19
----------------------------------------------------------------
```


```
Epoch 1
Train: Loss=1.9596 Batch_id=390 Accuracy=25.77: 100%|████████████████████████████████| 391/391 [00:31<00:00, 12.48it/s]
Test set: Average loss: 2.2579, Accuracy: 2417/10000 (24.17%)

Epoch 2
Train: Loss=1.7013 Batch_id=390 Accuracy=33.32: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.53it/s]
Test set: Average loss: 1.7278, Accuracy: 3678/10000 (36.78%)

Epoch 3
Train: Loss=1.5431 Batch_id=390 Accuracy=37.59: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.54it/s]
Test set: Average loss: 1.7659, Accuracy: 3879/10000 (38.79%)

Epoch 4
Train: Loss=1.7032 Batch_id=390 Accuracy=40.29: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.88it/s]
Test set: Average loss: 1.6077, Accuracy: 4172/10000 (41.72%)

Epoch 5
Train: Loss=1.5684 Batch_id=390 Accuracy=42.19: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.87it/s]
Test set: Average loss: 1.6524, Accuracy: 4060/10000 (40.60%)

Epoch 6
Train: Loss=1.6130 Batch_id=390 Accuracy=43.82: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.58it/s]
Test set: Average loss: 1.6448, Accuracy: 4354/10000 (43.54%)

Epoch 7
Train: Loss=1.4079 Batch_id=390 Accuracy=45.26: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.54it/s]
Test set: Average loss: 1.5400, Accuracy: 4668/10000 (46.68%)

Epoch 8
Train: Loss=1.4577 Batch_id=390 Accuracy=46.00: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.30it/s]
Test set: Average loss: 1.4488, Accuracy: 4870/10000 (48.70%)

Epoch 9
Train: Loss=1.5148 Batch_id=390 Accuracy=47.37: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.42it/s]
Test set: Average loss: 1.6286, Accuracy: 4572/10000 (45.72%)

Epoch 10
Train: Loss=1.4925 Batch_id=390 Accuracy=47.94: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.52it/s]
Test set: Average loss: 1.3862, Accuracy: 4993/10000 (49.93%)

Epoch 11
Train: Loss=1.3065 Batch_id=390 Accuracy=48.83: 100%|████████████████████████████████| 391/391 [00:28<00:00, 13.58it/s]
Test set: Average loss: 1.2743, Accuracy: 5432/10000 (54.32%)

Epoch 12
Train: Loss=1.5449 Batch_id=390 Accuracy=49.12: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.21it/s]
Test set: Average loss: 1.3890, Accuracy: 5133/10000 (51.33%)

Epoch 13
Train: Loss=1.5418 Batch_id=390 Accuracy=49.78: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.24it/s]
Test set: Average loss: 1.3006, Accuracy: 5391/10000 (53.91%)

Epoch 14
Train: Loss=1.3398 Batch_id=390 Accuracy=50.38: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.25it/s]
Test set: Average loss: 1.2214, Accuracy: 5621/10000 (56.21%)

Epoch 15
Train: Loss=1.3312 Batch_id=390 Accuracy=50.46: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.30it/s]
Test set: Average loss: 1.2928, Accuracy: 5344/10000 (53.44%)

Epoch 16
Train: Loss=1.3046 Batch_id=390 Accuracy=51.06: 100%|████████████████████████████████| 391/391 [00:30<00:00, 12.91it/s]
Test set: Average loss: 1.2305, Accuracy: 5584/10000 (55.84%)

Epoch 17
Train: Loss=1.2010 Batch_id=390 Accuracy=51.55: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.29it/s]
Test set: Average loss: 1.2045, Accuracy: 5640/10000 (56.40%)

Epoch 18
Train: Loss=1.4282 Batch_id=390 Accuracy=51.78: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.30it/s]
Test set: Average loss: 1.1356, Accuracy: 5934/10000 (59.34%)

Epoch 19
Train: Loss=1.3935 Batch_id=390 Accuracy=51.90: 100%|████████████████████████████████| 391/391 [00:29<00:00, 13.12it/s]
Test set: Average loss: 1.2086, Accuracy: 5677/10000 (56.77%)

Epoch 20
Train: Loss=1.3880 Batch_id=390 Accuracy=52.29: 100%|████████████████████████████████| 391/391 [00:30<00:00, 13.02it/s]
Test set: Average loss: 1.1859, Accuracy: 5780/10000 (57.80%)


```


## GN-8
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
         GroupNorm-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
         Dropout2d-4           [-1, 16, 32, 32]               0
         ConvLayer-5           [-1, 16, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]           4,608
         GroupNorm-7           [-1, 32, 32, 32]              64
              ReLU-8           [-1, 32, 32, 32]               0
         Dropout2d-9           [-1, 32, 32, 32]               0
        ConvLayer-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 16, 32, 32]             512
        MaxPool2d-12           [-1, 16, 16, 16]               0
       TransBlock-13           [-1, 16, 16, 16]               0
           Conv2d-14           [-1, 16, 16, 16]           2,304
        GroupNorm-15           [-1, 16, 16, 16]              32
             ReLU-16           [-1, 16, 16, 16]               0
        Dropout2d-17           [-1, 16, 16, 16]               0
        ConvLayer-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,304
        GroupNorm-20           [-1, 16, 16, 16]              32
             ReLU-21           [-1, 16, 16, 16]               0
        Dropout2d-22           [-1, 16, 16, 16]               0
        ConvLayer-23           [-1, 16, 16, 16]               0
           Conv2d-24           [-1, 32, 16, 16]           4,608
        GroupNorm-25           [-1, 32, 16, 16]              64
             ReLU-26           [-1, 32, 16, 16]               0
        Dropout2d-27           [-1, 32, 16, 16]               0
        ConvLayer-28           [-1, 32, 16, 16]               0
           Conv2d-29           [-1, 16, 16, 16]             512
        MaxPool2d-30             [-1, 16, 8, 8]               0
       TransBlock-31             [-1, 16, 8, 8]               0
           Conv2d-32             [-1, 16, 8, 8]           2,304
        GroupNorm-33             [-1, 16, 8, 8]              32
             ReLU-34             [-1, 16, 8, 8]               0
        Dropout2d-35             [-1, 16, 8, 8]               0
        ConvLayer-36             [-1, 16, 8, 8]               0
           Conv2d-37             [-1, 32, 8, 8]           4,608
        GroupNorm-38             [-1, 32, 8, 8]              64
             ReLU-39             [-1, 32, 8, 8]               0
        Dropout2d-40             [-1, 32, 8, 8]               0
        ConvLayer-41             [-1, 32, 8, 8]               0
           Conv2d-42             [-1, 64, 8, 8]          18,432
        GroupNorm-43             [-1, 64, 8, 8]             128
             ReLU-44             [-1, 64, 8, 8]               0
        Dropout2d-45             [-1, 64, 8, 8]               0
        ConvLayer-46             [-1, 64, 8, 8]               0
           Conv2d-47             [-1, 16, 8, 8]           1,024
        MaxPool2d-48             [-1, 16, 4, 4]               0
       TransBlock-49             [-1, 16, 4, 4]               0
AdaptiveAvgPool2d-50             [-1, 16, 1, 1]               0
           Conv2d-51             [-1, 10, 1, 1]             160
================================================================
Total params: 42,256
Trainable params: 42,256
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.02
Params size (MB): 0.16
Estimated Total Size (MB): 3.19
----------------------------------------------------------------
```
```
Epoch 1
Train: Loss=1.9748 Batch_id=657 Accuracy=21.41: 100%|████████████████████████████████| 658/658 [00:32<00:00, 20.43it/s]
Test set: Average loss: 1.9314, Accuracy: 2857/10000 (28.57%)

Epoch 2
Train: Loss=1.8826 Batch_id=657 Accuracy=29.06: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.15it/s]
Test set: Average loss: 1.8896, Accuracy: 3188/10000 (31.88%)

Epoch 3
Train: Loss=1.7196 Batch_id=657 Accuracy=31.76: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.29it/s]
Test set: Average loss: 1.7670, Accuracy: 3419/10000 (34.19%)

Epoch 4
Train: Loss=1.7452 Batch_id=657 Accuracy=34.60: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.19it/s]
Test set: Average loss: 1.7565, Accuracy: 3512/10000 (35.12%)

Epoch 5
Train: Loss=1.6431 Batch_id=657 Accuracy=36.54: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.55it/s]
Test set: Average loss: 1.6552, Accuracy: 3930/10000 (39.30%)

Epoch 6
Train: Loss=1.7956 Batch_id=657 Accuracy=37.89: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.83it/s]
Test set: Average loss: 1.6027, Accuracy: 4067/10000 (40.67%)

Epoch 7
Train: Loss=1.6279 Batch_id=657 Accuracy=39.03: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.72it/s]
Test set: Average loss: 1.6001, Accuracy: 4097/10000 (40.97%)

Epoch 8
Train: Loss=1.6325 Batch_id=657 Accuracy=40.03: 100%|████████████████████████████████| 658/658 [00:28<00:00, 23.07it/s]
Test set: Average loss: 1.5699, Accuracy: 4156/10000 (41.56%)

Epoch 9
Train: Loss=1.8395 Batch_id=657 Accuracy=41.44: 100%|████████████████████████████████| 658/658 [00:29<00:00, 21.95it/s]
Test set: Average loss: 1.5865, Accuracy: 4224/10000 (42.24%)

Epoch 10
Train: Loss=1.4163 Batch_id=657 Accuracy=42.22: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.03it/s]
Test set: Average loss: 1.5199, Accuracy: 4432/10000 (44.32%)

Epoch 11
Train: Loss=1.5222 Batch_id=657 Accuracy=42.82: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.77it/s]
Test set: Average loss: 1.4436, Accuracy: 4648/10000 (46.48%)

Epoch 12
Train: Loss=1.4563 Batch_id=657 Accuracy=43.67: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.78it/s]
Test set: Average loss: 1.4507, Accuracy: 4701/10000 (47.01%)

Epoch 13
Train: Loss=1.5724 Batch_id=657 Accuracy=44.30: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.58it/s]
Test set: Average loss: 1.4432, Accuracy: 4776/10000 (47.76%)

Epoch 14
Train: Loss=1.3720 Batch_id=657 Accuracy=44.97: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.53it/s]
Test set: Average loss: 1.4462, Accuracy: 4704/10000 (47.04%)

Epoch 15
Train: Loss=1.4768 Batch_id=657 Accuracy=45.69: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.67it/s]
Test set: Average loss: 1.3774, Accuracy: 5030/10000 (50.30%)

Epoch 16
Train: Loss=1.3991 Batch_id=657 Accuracy=45.91: 100%|████████████████████████████████| 658/658 [00:29<00:00, 21.95it/s]
Test set: Average loss: 1.4416, Accuracy: 4770/10000 (47.70%)

Epoch 17
Train: Loss=1.4126 Batch_id=657 Accuracy=46.47: 100%|████████████████████████████████| 658/658 [00:31<00:00, 21.01it/s]
Test set: Average loss: 1.3456, Accuracy: 5044/10000 (50.44%)

Epoch 18
Train: Loss=1.4778 Batch_id=657 Accuracy=46.99: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.63it/s]
Test set: Average loss: 1.3530, Accuracy: 5143/10000 (51.43%)

Epoch 19
Train: Loss=1.4597 Batch_id=657 Accuracy=48.03: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.38it/s]
Test set: Average loss: 1.3841, Accuracy: 5014/10000 (50.14%)

Epoch 20
Train: Loss=1.3177 Batch_id=657 Accuracy=48.00: 100%|████████████████████████████████| 658/658 [00:29<00:00, 21.99it/s]
Test set: Average loss: 1.3251, Accuracy: 5188/10000 (51.88%)

```


## GN-4
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
         GroupNorm-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
         Dropout2d-4           [-1, 16, 32, 32]               0
         ConvLayer-5           [-1, 16, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]           4,608
         GroupNorm-7           [-1, 32, 32, 32]              64
              ReLU-8           [-1, 32, 32, 32]               0
         Dropout2d-9           [-1, 32, 32, 32]               0
        ConvLayer-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 16, 32, 32]             512
        MaxPool2d-12           [-1, 16, 16, 16]               0
       TransBlock-13           [-1, 16, 16, 16]               0
           Conv2d-14           [-1, 16, 16, 16]           2,304
        GroupNorm-15           [-1, 16, 16, 16]              32
             ReLU-16           [-1, 16, 16, 16]               0
        Dropout2d-17           [-1, 16, 16, 16]               0
        ConvLayer-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,304
        GroupNorm-20           [-1, 16, 16, 16]              32
             ReLU-21           [-1, 16, 16, 16]               0
        Dropout2d-22           [-1, 16, 16, 16]               0
        ConvLayer-23           [-1, 16, 16, 16]               0
           Conv2d-24           [-1, 32, 16, 16]           4,608
        GroupNorm-25           [-1, 32, 16, 16]              64
             ReLU-26           [-1, 32, 16, 16]               0
        Dropout2d-27           [-1, 32, 16, 16]               0
        ConvLayer-28           [-1, 32, 16, 16]               0
           Conv2d-29           [-1, 16, 16, 16]             512
        MaxPool2d-30             [-1, 16, 8, 8]               0
       TransBlock-31             [-1, 16, 8, 8]               0
           Conv2d-32             [-1, 16, 8, 8]           2,304
        GroupNorm-33             [-1, 16, 8, 8]              32
             ReLU-34             [-1, 16, 8, 8]               0
        Dropout2d-35             [-1, 16, 8, 8]               0
        ConvLayer-36             [-1, 16, 8, 8]               0
           Conv2d-37             [-1, 32, 8, 8]           4,608
        GroupNorm-38             [-1, 32, 8, 8]              64
             ReLU-39             [-1, 32, 8, 8]               0
        Dropout2d-40             [-1, 32, 8, 8]               0
        ConvLayer-41             [-1, 32, 8, 8]               0
           Conv2d-42             [-1, 64, 8, 8]          18,432
        GroupNorm-43             [-1, 64, 8, 8]             128
             ReLU-44             [-1, 64, 8, 8]               0
        Dropout2d-45             [-1, 64, 8, 8]               0
        ConvLayer-46             [-1, 64, 8, 8]               0
           Conv2d-47             [-1, 16, 8, 8]           1,024
        MaxPool2d-48             [-1, 16, 4, 4]               0
       TransBlock-49             [-1, 16, 4, 4]               0
AdaptiveAvgPool2d-50             [-1, 16, 1, 1]               0
           Conv2d-51             [-1, 10, 1, 1]             160
================================================================
Total params: 42,256
Trainable params: 42,256
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.02
Params size (MB): 0.16
Estimated Total Size (MB): 3.19
----------------------------------------------------------------
```
```
Epoch 1
Train: Loss=2.0374 Batch_id=657 Accuracy=20.95: 100%|████████████████████████████████| 658/658 [00:31<00:00, 20.99it/s]
Test set: Average loss: 1.9165, Accuracy: 2897/10000 (28.97%)

Epoch 2
Train: Loss=1.8310 Batch_id=657 Accuracy=29.09: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.09it/s]
Test set: Average loss: 1.8348, Accuracy: 3217/10000 (32.17%)

Epoch 3
Train: Loss=1.8575 Batch_id=657 Accuracy=32.22: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.13it/s]
Test set: Average loss: 1.8479, Accuracy: 3212/10000 (32.12%)

Epoch 4
Train: Loss=1.7663 Batch_id=657 Accuracy=33.65: 100%|████████████████████████████████| 658/658 [00:29<00:00, 21.96it/s]
Test set: Average loss: 1.9100, Accuracy: 3076/10000 (30.76%)

Epoch 5
Train: Loss=1.7484 Batch_id=657 Accuracy=34.29: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.35it/s]
Test set: Average loss: 1.8003, Accuracy: 3497/10000 (34.97%)

Epoch 6
Train: Loss=1.6401 Batch_id=657 Accuracy=35.55: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.22it/s]
Test set: Average loss: 1.8506, Accuracy: 3452/10000 (34.52%)

Epoch 7
Train: Loss=1.7198 Batch_id=657 Accuracy=36.56: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.28it/s]
Test set: Average loss: 1.8173, Accuracy: 3459/10000 (34.59%)

Epoch 8
Train: Loss=1.8257 Batch_id=657 Accuracy=37.24: 100%|████████████████████████████████| 658/658 [00:27<00:00, 23.99it/s]
Test set: Average loss: 1.7616, Accuracy: 3642/10000 (36.42%)

Epoch 9
Train: Loss=1.6064 Batch_id=657 Accuracy=37.75: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.89it/s]
Test set: Average loss: 1.7810, Accuracy: 3592/10000 (35.92%)

Epoch 10
Train: Loss=1.5408 Batch_id=657 Accuracy=38.31: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.92it/s]
Test set: Average loss: 1.7982, Accuracy: 3588/10000 (35.88%)

Epoch 11
Train: Loss=1.6800 Batch_id=657 Accuracy=39.42: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.51it/s]
Test set: Average loss: 1.6859, Accuracy: 3798/10000 (37.98%)

Epoch 12
Train: Loss=1.6680 Batch_id=657 Accuracy=40.03: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.73it/s]
Test set: Average loss: 1.7766, Accuracy: 3713/10000 (37.13%)

Epoch 13
Train: Loss=1.6621 Batch_id=657 Accuracy=40.84: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.91it/s]
Test set: Average loss: 1.6370, Accuracy: 4027/10000 (40.27%)

Epoch 14
Train: Loss=1.5954 Batch_id=657 Accuracy=41.49: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.69it/s]
Test set: Average loss: 1.7963, Accuracy: 3511/10000 (35.11%)

Epoch 15
Train: Loss=1.5727 Batch_id=657 Accuracy=41.91: 100%|████████████████████████████████| 658/658 [00:28<00:00, 22.79it/s]
Test set: Average loss: 1.6618, Accuracy: 3987/10000 (39.87%)

Epoch 16
Train: Loss=1.5130 Batch_id=657 Accuracy=42.60: 100%|████████████████████████████████| 658/658 [00:29<00:00, 22.16it/s]
Test set: Average loss: 1.5441, Accuracy: 4301/10000 (43.01%)

Epoch 17
Train: Loss=1.5905 Batch_id=657 Accuracy=43.18: 100%|████████████████████████████████| 658/658 [00:31<00:00, 21.03it/s]
Test set: Average loss: 1.4906, Accuracy: 4519/10000 (45.19%)

Epoch 18
Train: Loss=1.6407 Batch_id=657 Accuracy=43.80: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.37it/s]
Test set: Average loss: 1.5229, Accuracy: 4470/10000 (44.70%)

Epoch 19
Train: Loss=1.6318 Batch_id=657 Accuracy=44.19: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.52it/s]
Test set: Average loss: 1.5150, Accuracy: 4459/10000 (44.59%)

Epoch 20
Train: Loss=1.4278 Batch_id=657 Accuracy=44.38: 100%|████████████████████████████████| 658/658 [00:30<00:00, 21.90it/s]
Test set: Average loss: 1.5137, Accuracy: 4569/10000 (45.69%)
```



## LN
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
         GroupNorm-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
         Dropout2d-4           [-1, 16, 32, 32]               0
         ConvLayer-5           [-1, 16, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]           4,608
         GroupNorm-7           [-1, 32, 32, 32]              64
              ReLU-8           [-1, 32, 32, 32]               0
         Dropout2d-9           [-1, 32, 32, 32]               0
        ConvLayer-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 16, 32, 32]             512
        MaxPool2d-12           [-1, 16, 16, 16]               0
       TransBlock-13           [-1, 16, 16, 16]               0
           Conv2d-14           [-1, 16, 16, 16]           2,304
        GroupNorm-15           [-1, 16, 16, 16]              32
             ReLU-16           [-1, 16, 16, 16]               0
        Dropout2d-17           [-1, 16, 16, 16]               0
        ConvLayer-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,304
        GroupNorm-20           [-1, 16, 16, 16]              32
             ReLU-21           [-1, 16, 16, 16]               0
        Dropout2d-22           [-1, 16, 16, 16]               0
        ConvLayer-23           [-1, 16, 16, 16]               0
           Conv2d-24           [-1, 32, 16, 16]           4,608
        GroupNorm-25           [-1, 32, 16, 16]              64
             ReLU-26           [-1, 32, 16, 16]               0
        Dropout2d-27           [-1, 32, 16, 16]               0
        ConvLayer-28           [-1, 32, 16, 16]               0
           Conv2d-29           [-1, 16, 16, 16]             512
        MaxPool2d-30             [-1, 16, 8, 8]               0
       TransBlock-31             [-1, 16, 8, 8]               0
           Conv2d-32             [-1, 16, 8, 8]           2,304
        GroupNorm-33             [-1, 16, 8, 8]              32
             ReLU-34             [-1, 16, 8, 8]               0
        Dropout2d-35             [-1, 16, 8, 8]               0
        ConvLayer-36             [-1, 16, 8, 8]               0
           Conv2d-37             [-1, 32, 8, 8]           4,608
        GroupNorm-38             [-1, 32, 8, 8]              64
             ReLU-39             [-1, 32, 8, 8]               0
        Dropout2d-40             [-1, 32, 8, 8]               0
        ConvLayer-41             [-1, 32, 8, 8]               0
           Conv2d-42             [-1, 64, 8, 8]          18,432
        GroupNorm-43             [-1, 64, 8, 8]             128
             ReLU-44             [-1, 64, 8, 8]               0
        Dropout2d-45             [-1, 64, 8, 8]               0
        ConvLayer-46             [-1, 64, 8, 8]               0
           Conv2d-47             [-1, 16, 8, 8]           1,024
        MaxPool2d-48             [-1, 16, 4, 4]               0
       TransBlock-49             [-1, 16, 4, 4]               0
AdaptiveAvgPool2d-50             [-1, 16, 1, 1]               0
           Conv2d-51             [-1, 10, 1, 1]             160
================================================================
Total params: 42,256
Trainable params: 42,256
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.02
Params size (MB): 0.16
Estimated Total Size (MB): 3.19
----------------------------------------------------------------
```

```
Epoch 1
Train: Loss=1.9026 Batch_id=781 Accuracy=20.08: 100%|████████████████████████████████| 782/782 [00:32<00:00, 23.75it/s]
Test set: Average loss: 2.3685, Accuracy: 1662/10000 (16.62%)

Epoch 2
Train: Loss=1.9175 Batch_id=781 Accuracy=27.76: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.89it/s]
Test set: Average loss: 2.0742, Accuracy: 2336/10000 (23.36%)

Epoch 3
Train: Loss=1.7862 Batch_id=781 Accuracy=30.01: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.52it/s]
Test set: Average loss: 2.2317, Accuracy: 2017/10000 (20.17%)

Epoch 4
Train: Loss=1.8477 Batch_id=781 Accuracy=31.93: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.81it/s]
Test set: Average loss: 2.3689, Accuracy: 2034/10000 (20.34%)

Epoch 5
Train: Loss=2.4706 Batch_id=781 Accuracy=33.51: 100%|████████████████████████████████| 782/782 [00:30<00:00, 26.06it/s]
Test set: Average loss: 2.1540, Accuracy: 2213/10000 (22.13%)

Epoch 6
Train: Loss=1.7707 Batch_id=781 Accuracy=34.37: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.39it/s]
Test set: Average loss: 2.1985, Accuracy: 2340/10000 (23.40%)

Epoch 7
Train: Loss=1.7581 Batch_id=781 Accuracy=35.81: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.94it/s]
Test set: Average loss: 2.6776, Accuracy: 1938/10000 (19.38%)

Epoch 8
Train: Loss=1.4394 Batch_id=781 Accuracy=36.89: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.48it/s]
Test set: Average loss: 2.0185, Accuracy: 2839/10000 (28.39%)

Epoch 9
Train: Loss=1.6684 Batch_id=781 Accuracy=37.43: 100%|████████████████████████████████| 782/782 [00:29<00:00, 26.32it/s]
Test set: Average loss: 1.9698, Accuracy: 3050/10000 (30.50%)

Epoch 10
Train: Loss=1.6523 Batch_id=781 Accuracy=38.10: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.89it/s]
Test set: Average loss: 2.1354, Accuracy: 2540/10000 (25.40%)

Epoch 11
Train: Loss=2.1931 Batch_id=781 Accuracy=38.99: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.75it/s]
Test set: Average loss: 1.8910, Accuracy: 3096/10000 (30.96%)

Epoch 12
Train: Loss=1.5405 Batch_id=781 Accuracy=39.73: 100%|████████████████████████████████| 782/782 [00:30<00:00, 26.00it/s]
Test set: Average loss: 1.8742, Accuracy: 3149/10000 (31.49%)

Epoch 13
Train: Loss=1.2288 Batch_id=781 Accuracy=40.12: 100%|████████████████████████████████| 782/782 [00:29<00:00, 26.11it/s]
Test set: Average loss: 2.2647, Accuracy: 2437/10000 (24.37%)

Epoch 14
Train: Loss=1.4766 Batch_id=781 Accuracy=40.47: 100%|████████████████████████████████| 782/782 [00:29<00:00, 26.22it/s]
Test set: Average loss: 2.0615, Accuracy: 2924/10000 (29.24%)

Epoch 15
Train: Loss=1.3572 Batch_id=781 Accuracy=41.15: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.93it/s]
Test set: Average loss: 2.1267, Accuracy: 2950/10000 (29.50%)

Epoch 16
Train: Loss=1.5996 Batch_id=781 Accuracy=41.92: 100%|████████████████████████████████| 782/782 [00:30<00:00, 26.06it/s]
Test set: Average loss: 1.9216, Accuracy: 3207/10000 (32.07%)

Epoch 17
Train: Loss=1.4838 Batch_id=781 Accuracy=41.97: 100%|████████████████████████████████| 782/782 [00:30<00:00, 25.96it/s]
Test set: Average loss: 1.7734, Accuracy: 3533/10000 (35.33%)

Epoch 18
Train: Loss=1.6567 Batch_id=781 Accuracy=42.51: 100%|████████████████████████████████| 782/782 [00:29<00:00, 26.39it/s]
Test set: Average loss: 1.9904, Accuracy: 3326/10000 (33.26%)

Epoch 19
Train: Loss=1.4070 Batch_id=781 Accuracy=42.41: 100%|████████████████████████████████| 782/782 [00:29<00:00, 26.33it/s]
Test set: Average loss: 1.8574, Accuracy: 3444/10000 (34.44%)

Epoch 20
Train: Loss=1.4742 Batch_id=781 Accuracy=43.31: 100%|████████████████████████████████| 782/782 [00:29<00:00, 26.13it/s]
Test set: Average loss: 1.9554, Accuracy: 3227/10000 (32.27%)

```