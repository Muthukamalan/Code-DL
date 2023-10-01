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

def plot_data(loader,classes):
    inputs,targets = next(iter(loader))
    fig = plt.figure()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.tight_layout()

        # unnormalize = Normalize( (-mean/std) , (1/std) )
        unnormizae  = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(inputs[i])

        plt.imshow(transforms.ToPILImage()(unnormizae))
        plt.title(
            classes[targets[i].item()]
        )
        plt.xticks([]);plt.yticks([])



def plot_model_training_curves(train_accs, test_accs, train_losses, test_losses):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accs)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title("Test Accuracy")
    plt.plot()


def plot_confusion_matrix(labels,preds,classes,normalize=True):
    mat = ConfusionMatrix(task="multiclass",num_classes=10)
    mat = mat(preds,labels).numpy()
    if normalize:
        df_mat = pd.DataFrame(
                        mat/np.sum(mat,axis=1)[:,None],
                        index=[i for i in classes],
                        columns=[i for i in classes]
                )
    else:
        df_mat = pd.DataFrame(mat,index=[i for i in classes],columns=[i for i in classes])
    plt.figure(figsize=(7,5))
    sns.heatmap(df_mat,annot=True,cmap='Blues',fmt='.3f',linewidths=0.5)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def get_all_and_incorrect_preds(model, loader, device):
    incorrect = []
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1).cpu()
            target = target.cpu()
            all_preds = torch.cat((all_preds, pred), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append((d.cpu(), t, p, o[p.item()].cpu()))

    return all_preds, all_labels, incorrect

def plot_incorrect_preds(incorrect, classes):
    # incorrect (data, target, pred, output)
    print(f"Total Incorrect Predictions {len(incorrect)}")
    fig = plt.figure(figsize=(10, 5))
    plt.suptitle("Target | Predicted Label")
    for i in range(10):
        plt.subplot(2, 5, i + 1, aspect="auto")

        # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(incorrect[i][0])
        plt.imshow(transforms.ToPILImage()(unnormalized))
        plt.title(
            f"{classes[incorrect[i][1].item()]}|{classes[incorrect[i][2].item()]}",
            # fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])





