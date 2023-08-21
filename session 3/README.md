# High-Level Overview of Pytorch Framework

- Tensor ( torch.Tensor )
- Dataset and DataLoaders ( torch.utils.data )
- Autograd ( torch.autograd )
- Model Building ( torch.nn.Module / torch.nn.Sequential )
- Transforms (torchvision.transforms )
- Optimization ( torch.optim )

# Torch Useful Utilities
- Saving and loading model
- checkpointing
- Debugging and visualizing the models with tensorboard
- Distributed training

# Torch Libraries
- Torchvision
- Torchtext
- Torchserve
- Torchdata
- Torchaudio
- Torchrec

# Torch Ecosystem
- Transformers
- PyTorchNLP
- Fastai
- PyTorch3D
- PyTorch Video
-


# Terminology

**1 epoch**: one complete pass of the training dataset through the algorithms

**batch_size**:
- number of training_examples in one forward/backward pass.
- more number examples per pass results more memory

**no of batches**:  number of passes, each pass using batch_size number of example
            