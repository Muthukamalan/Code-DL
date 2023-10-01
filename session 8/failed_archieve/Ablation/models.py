import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class ConvLayer(nn.Module):
    def __init__(self,inc:int,outc:int,k:int,p:int,norm:str,dp_rate:int,grp:int=0):
        super(ConvLayer,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=3,padding=p,bias=False),
            self.get_norm(norm=norm,grp=grp,num_f=outc),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dp_rate)
        )

    def get_norm(self,norm:str,num_f:int,grp:int=0):
        if norm=='bn':
            return nn.BatchNorm2d(num_features=num_f)
        elif norm=='ln':
            return nn.GroupNorm(num_groups=1,num_channels=num_f)
        elif norm=='gn':
            return nn.GroupNorm(num_groups=grp,num_channels=num_f)
        else:
            raise ValueError("choose bn/ln/gn")

    def forward(self,x):
        x = self.layer(x)
        return x


class TransBlock(nn.Module):
    def __init__(self,inc:int,outc:int):
        super(TransBlock,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=1,bias=False),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
    def forward(self,x):
        x = self.layer(x)
        return x
class Net(nn.Module):
        def __init__(self,norm_method:str,grp:int=0,dp_rate=0.1):
            '''channels:[
                        3,    # INPUT
                        16,   # output_layer-C4, C5, C6-input_layer
                        20,   # output_layer-C6, C7-input_layer
                        28,   # output_layer-C7, C8, C9
                        40,   # GAP
                        10    # OUTPUT
            ]'''
            super(Net,self).__init__()
            if not norm_method in('bn','ln','gn'):
                raise ValueError('choose bn/ln/gn')
            else:
                self.norm = norm_method
                if self.norm =='gn':
                    self.grp=grp
                elif self.norm=='ln':
                    self.grp=1
                elif self.norm=='bn':
                    self.grp=1

            self.dp_rate=dp_rate
            
            self.C1 = ConvLayer(inc=3,outc=16,k=3,p=1,norm=self.norm,dp_rate=self.dp_rate)
            self.C2 = ConvLayer(inc=16,outc=16,k=3,p=1,norm=self.norm,dp_rate=self.dp_rate)
            self.t1 = TransBlock(inc=16,outc=16)
            
            self.C4 = ConvLayer(inc=16,outc=20,k=3,p=1,norm=self.norm,dp_rate=self.dp_rate)
            self.C5 = ConvLayer(inc=20,outc=20,k=3,p=1,norm=self.norm,dp_rate=self.dp_rate)
            self.t2 = TransBlock(inc=20,outc=28)

            self.C7 = ConvLayer(inc=28,outc=40,p=1,k=3,norm=self.norm,dp_rate=self.dp_rate)
            self.C8 = ConvLayer(inc=40,outc=40,p=1,k=3,norm=self.norm,dp_rate=self.dp_rate)
            self.C9 = ConvLayer(inc=40,outc=40,p=1,k=3,norm=self.norm,dp_rate=self.dp_rate)

            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.out = nn.Conv2d(in_channels=40,out_channels=10,kernel_size=1,stride=1,bias=False)
    
        def forward(self,x):
            x = self.C1(x)
            x = x + self.C2(x)
            x = self.t1(x)

            x = self.C4(x)
            x = x + self.C5(x)
            x = self.t2(x)

            x = self.C7(x)
            x = x + self.C8(x)
            x = x + self.C9(x)
            x = self.gap(x)
            x = self.out(x)
            return F.log_softmax(x.view(-1,10), dim=1)