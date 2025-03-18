import torch.nn as nn

class DCNv2(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim, cross_layers, deep_layers, onehot_size):
        super(DCNv2, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cat_card + 1, embedding_dim) for cat_card in cat_cardinalities])
        input_dim = num_numerical + len(cat_cardinalities) * embedding_dim + onehot_size
        self.cross_net = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(cross_layers)])
        self.deep_net = nn.Sequential(
            *[nn.Linear(input_dim, deep_layers[0]), nn.ReLU()] +
            sum([[nn.Linear(deep_layers[i], deep_layers[i+1]), nn.ReLU()] for i in range(len(deep_layers)-1)], [])
        )
        self.output_layer = nn.Linear(deep_layers[-1], 1)

    def forward(self, numerical, categorical_emb, categorical_onehot):
        cat_embeds = [emb(categorical_emb[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)
        x = torch.cat([numerical, cat_embeds, categorical_onehot], dim=1)
        for layer in self.cross_net:
            x = x + layer(x)
        x = self.deep_net(x)
        return torch.sigmoid(self.output_layer(x)).squeeze(1)
