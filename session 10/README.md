# Objective
```
Write a custom ResNet architecture for CIFAR10 that has the following architecture:
    - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    - Layer1:
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        Add(X, R1)
    - Layer 2:
        Conv 3x3 [256k]
        MaxPooling2D
        BN
        ReLU
    - Layer 3
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        Add(X, R2)
    - MaxPooling with Kernel Size 4
    - FC Layer 
    - SoftMax

Uses One Cycle Policy such that:
    - Total Epochs = 24
    - Max at Epoch = 5
    - LRMIN = FIND
    - LRMAX = FIND
    - NO Annihilation

Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
Batch size = 512
Use ADAM, and CrossEntropyLoss
Target Accuracy: 90%
```
[custom model reference](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/)

 

 

 

