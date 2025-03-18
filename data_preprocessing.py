import torch
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(numerical_data, categorical_emb_data, categorical_onehot_data, labels, batch_size=32):
    dataset = TensorDataset(numerical_data, categorical_emb_data, categorical_onehot_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_val_split(numerical_data, categorical_emb_data, categorical_onehot_data, labels, split_ratio=0.8):
    N = len(numerical_data)
    split_idx = int(N * split_ratio)

    train_data = (numerical_data[:split_idx], categorical_emb_data[:split_idx], categorical_onehot_data[:split_idx], labels[:split_idx])
    val_data = (numerical_data[split_idx:], categorical_emb_data[split_idx:], categorical_onehot_data[split_idx:], labels[split_idx:])

    return train_data, val_data
