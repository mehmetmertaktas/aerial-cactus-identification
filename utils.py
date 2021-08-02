import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import torch
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score
from parameters import DEVICE

def init_weights(layer):
    if type(layer) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

def display_data(data, n=10, classes=None):
    fig, ax = plt.subplots(1, n, figsize=(15,3))
    indices = np.random.randint(0, len(data), size=n)
    low = min([data[i][0].min() for i in indices])
    high = max([data[i][0].max() for i in indices])
    for i, j in enumerate(indices):
        ax[i].imshow(np.transpose((data[j][0] - low) / (high - low), (1, 2, 0)))
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[data[j][1]])
    plt.show()

def train_epoch(model,
                dataloader,
                lr=1e-3,
                optimizer=None,
                loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss, accuracy, count = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss
        predicted = torch.max(out, 1)[1]
        accuracy += (predicted == y).sum()
        count += len(y)
    return total_loss.item() / count, accuracy.item() / count

def validate(model,
             dataloader,
             loss_fn=nn.NLLLoss()):
    model.eval()
    total_loss, accuracy, count = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            total_loss += loss_fn(out, y)
            predicted = torch.max(out, 1)[1]
            accuracy += (predicted == y).sum()
            count += len(y)
    return total_loss.item() / count, accuracy.item() / count

def train(model,
          train_loader,
          valid_loader=None,
          optimizer=None,
          lr=1e-3,
          epochs=10,
          loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    history = {'train_loss': [], 
               'train_accuracy': []}
    if valid_loader is not None:
        history['validation_loss'] = []
        history['validation_accuracy'] = []
    for epoch in range(epochs):
        tl, ta = train_epoch(model,
                             train_loader,
                             lr=lr,
                             optimizer=optimizer,
                             loss_fn=loss_fn)
        history['train_loss'].append(tl)
        history['train_accuracy'].append(ta)
        if valid_loader is not None:
            vl, va = validate(model, valid_loader, loss_fn=loss_fn)
            print(f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Val Acc = {va:.3f}, Train Loss = {tl:.3f}, Val Loss={vl:.3f}")
            history['validation_loss'].append(vl)
            history['validation_accuracy'].append(va)
        else:
            print(f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Train Loss = {tl:.3f}")
    return history

def plot_auc_curve(model, dataloader):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_true.append(int(y))
            out = model(X)
            y_score.append(float(torch.exp(out)[0][1]))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    textstr = f'AUC score = {auc_score:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, 
             textstr, fontsize=14, 
             verticalalignment='top', bbox=props)
    plt.plot(fpr, tpr)
    plt.show()
    return fpr, tpr, thresholds

def plot_history(history, validation=False):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history['train_accuracy'], label='Training')
    if validation:
        plt.plot(history['validation_accuracy'], label='Validation')
    plt.legend()
    plt.subplot(122)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(history['train_loss'], label='Training')
    if validation:
        plt.plot(history['validation_loss'], label='Validation')
    plt.legend()
    plt.show()

def save_results(path, dataloader, model):
    model.eval()
    result = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y[0]
            out = model(X)
            result.append([y, float(torch.exp(out)[0][1])])
    df = pd.DataFrame(result, columns = ['id', 'has_cactus'])
    df = df.set_index('id')
    df = df.sort_values('id')
    df.to_csv(path)
    return df
