import optuna
from train import train_model, evaluate_model
from model import DCNv2

def train_evaluate_model(trial, train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size):
    embedding_dim = trial.suggest_int('embedding_dim', 4, 16)
    cross_layers = trial.suggest_int('cross_layers', 1, 3)
    deep_layers = [trial.suggest_int(f'deep_layer_{i}', 16, 64) for i in range(2)]

    model = DCNv2(num_numerical, cat_cardinalities, embedding_dim, cross_layers, deep_layers, onehot_size)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True))
    criterion = nn.BCELoss()

    train_model(model, train_loader, optimizer, criterion, num_epochs=20)
    auc = evaluate_model(model, val_loader)

    return auc

def optimize_hyperparameters(train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size, n_trials=50):
    def objective(trial):
        auc = train_evaluate_model(trial, train_loader, val_loader, num_numerical, cat_cardinalities, onehot_size)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study
