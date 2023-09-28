import torch
from typing import List
from torch import Tensor
from utils import get_device

device = get_device()

class Tester:
    def __init__(self,model,device,test_loader,criterion) -> None:
        self.test_losses = []
        self.test_accuracies = []
        if device=='cuda':
            self.model = model.cuda()
        else:
            self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def test(self):

        # Keep model in Test
        self.model.eval()

        test_loss=0
        correct=0

        # no backpropagation
        with torch.no_grad():
            for inputs_,targets in self.test_loader:
                inputs_ = inputs_.to(self.device)
                targets = targets.to(self.device)

                # prediction
                outputs = self.model(inputs_)
                #calc loss
                loss = self.criterion(outputs,targets)

                test_loss+=loss.item()

                # get the index of the max log-probability
                pred = outputs.argmax(dim=1,keepdim=True)

                #
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss/=len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print(
            f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)}= {100.0* correct/len(self.test_loader.dataset)}"
        )
        self.test_accuracies.append(100 * correct/len(self.test_loader.dataset))
        return 100.0* correct/len(self.test_loader.dataset),test_loss
    
    def get_misclassified_images(self):
        self.model.eval()

        images: List[Tensor] = []
        predictions          = []
        labels               = []

        with torch.no_grad():
            for inputs,targets in self.test_loader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                _,pred = torch.max(outputs,1)

                for i in range(len(pred)):
                    if pred[i]!=targets[i]:
                        images.append(inputs[i])
                        predictions.append(pred[i])
                        labels.append(targets[i])
        return images,predictions,labels