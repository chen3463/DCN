import numpy as np

def build_feature_names(num_numerical, emb_cat_cardinalities, onehot_cat_names, onehot_cat_dims):
    num_feature_names = [f"num_{i}" for i in range(num_numerical)]

    emb_feature_names = []
    emb_feature_slices = []
    start = num_numerical
    for i, cardinality in enumerate(emb_cat_cardinalities):
        emb_feature_names.append(f"emb_cat_{i}")
        emb_feature_slices.append((start, start + 1))
        start += 1

    onehot_feature_names = []
    onehot_feature_slices = []
    for i, name in enumerate(onehot_cat_names):
        dim = onehot_cat_dims[i]
        onehot_feature_names.append(name)
        onehot_feature_slices.append((start, start + dim))
        start += dim

    feature_names = num_feature_names + emb_feature_names + onehot_feature_names

    return feature_names, emb_feature_slices, onehot_feature_slices
