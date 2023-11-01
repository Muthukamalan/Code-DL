import numpy as np
from torch import nn
from torch import functional as F
import torch

class DepthwiseConvLayer(nn.Module):
    '''
        Depthwise Conv + Pointwise Conv = Depthwise Seperable COnv 
        depthwise have p=p and pointwiseConv has no padding to adjust total padding
    '''
    def __init__(self,inc,outc,s,p,dp_rate)->None:
        super(DepthwiseConvLayer,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=inc,groups=inc,kernel_size=3,stride=s,padding=p,bias=False, padding_mode='reflect'),
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=1,padding=0,bias=False, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(outc),
            nn.Dropout2d(dp_rate)
        )
    def forward(self,x):
        return self.layer(x)


class ConvLayer(nn.Module):
    def __init__(
        self,
        inc:int,
        outc:int,
        k:int,
        p:int,
        s:int,
        d:int,
        dp_rate:int
    ):
        super(ConvLayer,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=3,padding=p,bias=False,dilation=d,stride=s, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=outc),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dp_rate)
        )
    def forward(self,x):
        x = self.layer(x)
        return x


class TransBlock(nn.Module):
    def __init__(
        self,
        inc:int,
        outc:int,
        p:int,
        s:int
    ):
        super(TransBlock,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=1,stride=s,padding=p,bias=False, padding_mode='reflect'),
        )
    def forward(self,x):
        x = self.layer(x)
        return x