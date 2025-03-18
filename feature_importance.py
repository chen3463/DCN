import shap
import numpy as np
import pandas as pd
import torch

def feature_importance(model, data_loader, feature_names, emb_feature_slices, onehot_feature_slices):
    model.eval()

    numerical_data, categorical_emb_data, categorical_onehot_data, _ = next(iter(data_loader))
    input_data = torch.cat([numerical_data, categorical_emb_data.float(), categorical_onehot_data], dim=1)
    input_data_np = input_data.cpu().numpy()

    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            num = x_tensor[:, :numerical_data.shape[1]]
            emb = x_tensor[:, numerical_data.shape[1]:numerical_data.shape[1] + categorical_emb_data.shape[1]].long()
            onehot = x_tensor[:, numerical_data.shape[1] + categorical_emb_data.shape[1]:]
            return model(num, emb, onehot).squeeze().cpu().numpy()

    masker = shap.maskers.Independent(input_data_np)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(input_data_np)

    mean_shap = np.abs(shap_values.values).mean(axis=0)

    mean_shap_values = []

    # Numerical
    num_features = numerical_data.shape[1]
    mean_shap_values.extend(mean_shap[:num_features])

    # Embeddings (aggregate per variable)
    for start, end in emb_feature_slices:
        if start < end:
            mean_value = mean_shap[start:end].mean()
        else:
            mean_value = 0.0
        mean_shap_values.append(mean_value)

    # One-hot (aggregate per variable)
    for start, end in onehot_feature_slices:
        if start < end:
            mean_value = mean_shap[start:end].mean()
        else:
            mean_value = 0.0
        mean_shap_values.append(mean_value)

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap_value': mean_shap_values
    }).sort_values(by='mean_abs_shap_value', ascending=False)

    return df
