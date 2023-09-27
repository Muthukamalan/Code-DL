from tqdm import tqdm 

class Trainer:
    def __init__(self,model,train_loader,optimizer,criterion,device) -> None:
        self.train_losses = []
        self.train_accuracies=[]
        self.epoch_train_accuracies=[]
        self.model = model.to(device)
        self.train_loader= train_loader
        self.optimizer = optimizer
        self.criterion =criterion
        self.device = device
        self.lr_history =[]

    def train(self,epoch,use_l1,lambda_l1=0.01):

        self.model.train()

        lr_trend =[]
        correct = 0
        processed=0
        train_loss=0

        # output looks nicer
        pbar = tqdm(self.train_loader)

        for batch_id,(inputs,targets) in enumerate(pbar):
            #transfer to device
            inputs = inputs.to(self.device)
            targets= targets.to(self.device)

            # set grad=0
            self.optimizer.zero_grad()

            # prediction
            outputs = self.model(inputs)

            # calculate loss
            loss = self.criterion(outputs,targets)

            # use L1 loss = Actual loss + Parameters Count
            l1=0
            if use_l1:
                for p in self.model.parameters():
                    l1+=p.abs().sum()
            loss = loss+lambda_l1*l1

            # Plotting train_loss
            self.train_losses.append(loss.item())

            # backpropagation
            loss.backward()
            self.optimizer.step()

            # get the index of the max log-probability
            pred = outputs.argmax(dim=1,keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            processed+=len(inputs)


            # Description
            pbar.set_description(
                desc= f"Batch={batch_id} | Epoch={epoch} | LR={self.optimizer.param_groups[0]['lr']} | Loss={loss.item():3.4f} | Accuracy={100*correct/processed:0.4f}"
            )
            self.train_accuracies.append(100*correct/processed)

            # After all the batches are done append accuracy for epoch
            self.epoch_train_accuracies.append(100*correct/processed)

            self.lr_history.extend(lr_trend)
            return (
                100 * correct/processed , 
                train_loss/len(self.train_loader), 
                lr_trend
            )
