import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torchmetrics import ConfusionMatrix
from torchvision import transforms


def plot_class_distribution(loader,classes):
    class_counts={}
    for cname in classes:
        class_counts[cname]=0
    for _,labels in loader:
        for lbl in labels:
            class_counts[
                    classes[ lbl.item() ]
            ]+=1
    fig = plt.figure()
    plt.suptitle('Class Distribution')
    plt.bar(
        range( len(class_counts) ),
        list(class_counts.values())
    )
    plt.xticks(
        range( len(class_counts) ),
        list(class_counts.keys()),
        rotation=90
     )
    plt.tight_layout()
    plt.show()

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model,test_loader,device,classes):
    model.eval()
    predictions = []
    labels = []  # TRUTH
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(target.cpu().numpy())
    
    cm= confusion_matrix(y_true= [i.item() for i in labels],y_pred=[i.item() for i in predictions])
    columns ={}
    for i,v in enumerate(classes):
        columns[i]=v
    plt.figure(figsize=(9,6))
    sns.heatmap(pd.DataFrame(cm).rename(columns=columns,index=columns),annot=True,fmt='',cmap="crest")
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.show()


def plot_curves(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()