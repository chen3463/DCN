import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for numerical, categorical_emb, categorical_onehot, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(numerical, categorical_emb, categorical_onehot).squeeze()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

def evaluate_model(model, val_loader):
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for numerical, categorical_emb, categorical_onehot, labels in val_loader:
            outputs = model(numerical, categorical_emb, categorical_onehot).squeeze()
            val_preds.extend(outputs.numpy())
            val_labels.extend(labels.squeeze().numpy())
    auc = roc_auc_score(val_labels, val_preds)
    return auc
