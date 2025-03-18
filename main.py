import torch
import numpy as np
from model import DCNv2
from data_preprocessing import create_dataloader, train_val_split
from hyperparameter_optimization import optimize_hyperparameters
from utils import build_feature_names
from feature_importance import feature_importance

# Dummy data
num_numerical = 5
num_categorical_emb = 3
onehot_cardinalities = [4, 3]
onehot_size = sum(onehot_cardinalities)
N = 1000

numerical_data = torch.randn(N, num_numerical)
categorical_emb_data = torch.randint(0, 10, (N, num_categorical_emb))
categorical_onehot_data = torch.randint(0, 2, (N, onehot_size))
labels = torch.randint(0, 2, (N, 1)).float()

# Train/val split
train_data, val_data = train_val_split(numerical_data, categorical_emb_data, categorical_onehot_data, labels)
train_loader = create_dataloader(*train_data)
val_loader = create_dataloader(*val_data)

# Hyperparameter optimization
study = optimize_hyperparameters(train_loader, val_loader, num_numerical, [10, 10, 10], onehot_size)

best_params = study.best_trial.params
embedding_dim = best_params['embedding_dim']
cross_layers = best_params['cross_layers']
deep_layers = [best_params['deep_layer_0'], best_params['deep_layer_1']]
learning_rate = best_params['lr']

# Instantiate the final model
final_model = DCNv2(num_numerical, [10, 10, 10], embedding_dim, cross_layers, deep_layers, onehot_size)
optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()

# Train final model
final_model.train()
train_model(final_model, train_loader, optimizer, criterion)

# Build feature names
onehot_cat_names = [f"cat{i+1}" for i in range(len(onehot_cardinalities))]
feature_names, emb_feature_slices, onehot_feature_slices = build_feature_names(
    num_numerical, [10, 10, 10], onehot_cat_names, [4, 3]
)

# Calculate and print feature importance
df_importance = feature_importance(final_model, train_loader, feature_names, emb_feature_slices, onehot_feature_slices)
print(df_importance)
