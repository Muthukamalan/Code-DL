class Useless:
    def __init__(self,dataset,label) -> None:
        self.dataset = dataset
        self.label   = label

    def __len__(self)-> int:
        return len(self.dataset)
    

class Useless2:
    def __init__(self,dataset,label) -> None:
        self.dataset= dataset
        self.label  = label

    def __len__(self)->int:
        return len(self.dataset)
    
    def __getitem__(self,idx)-> tuple:
        return (self.dataset[idx],self.label[idx])





'''
import torch
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self,data,label)->None:
        self.data = data
        self.label= label
    
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self,idx)->tuple:
        return (self.data[idx],self.label[idx])
    
my_datasets = Dataset(data,label)
my_dataloader = DataLoader(my_datasets,batch_size=128,shuffle=True)


'''

if __name__=='__main__':
    dataset = ['bug','butterfly','glass','latop','dog']
    label   = [ 1   ,  1        , 0     , 0     , 1]

    u = Useless(dataset=dataset,label=label)
    print(len(u))

    v = Useless2(dataset=dataset,label=label)
    
    for d,l in v:
        print(d,l)
    print(v[2])
    print(len(v))

    
