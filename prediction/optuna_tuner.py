import json
import os
import random
import numpy as np
import torch
import optuna
from optuna.exceptions import TrialPruned
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from prediction.predictor import DeepBERPredictor
from load_set import load_csv_dataset
from dataset_manipulation import extend_features, exclude_columns


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(input_size, hidden_sizes, batch_norm, dropout, activation_name="relu", logBER=True):
    # Map activation name to a PyTorch activation module
    activation_map = {
        "relu": torch.nn.ReLU(),
        "leaky_relu": torch.nn.LeakyReLU(negative_slope=0.01),
        "elu": torch.nn.ELU(),
        "gelu": torch.nn.GELU(),
    }
    activation_fn = activation_map.get(activation_name, torch.nn.ReLU())

    return DeepBERPredictor(
        input_size=input_size,
        hidden=hidden_sizes,
        activation_fn=activation_fn,
        logBER=logBER,
        batch_norm=batch_norm,
        dropout=dropout,
    )


def build_fold_loaders(x_array, y_array, train_idx, val_idx, batch_size, batch_norm):
    x_train = x_array[train_idx]
    y_train = y_array[train_idx]
    x_val = x_array[val_idx]
    y_val = y_array[val_idx]

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)
    train_std = np.where(train_std == 0.0, 1.0, train_std)

    x_train = ((x_train - train_mean) / train_std).astype(np.float32)
    x_val = ((x_val - train_mean) / train_std).astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    effective_batch_size = min(batch_size, len(train_idx))
    if effective_batch_size < 1:
        raise ValueError("Training fold is empty.")

    if batch_norm and effective_batch_size < 2:
        raise ValueError("Batch normalization requires at least 2 samples per training batch.")

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=batch_norm,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_idx)) if len(val_idx) > 0 else 1,
        shuffle=False,
    )

    return train_loader, val_loader


def run_optuna(x_array, y_array, feature_columns, gray_area_interval=None, n_trials=20, n_epochs=5, seed=42, study_name="deepber_optuna", storage=None, cv_folds=3):
    set_seed(seed)

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")

    print(f"[optuna] Loaded dataset: x_shape={x_array.shape}, y_shape={y_array.shape}, features={len(feature_columns)}")

    # Apply label filtering if gray_area_interval is provided
    if gray_area_interval is not None:
        lower_thres, upper_thres = gray_area_interval
        label_mask = (y_array >= lower_thres) & (y_array <= upper_thres)
        x_array = x_array[label_mask]
        y_array = y_array[label_mask]
        print(f"[optuna] Dataset: x_shape={x_array.shape}, features={len(feature_columns)}, samples={len(y_array)}")

    input_size = len(feature_columns)
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    print(f"[optuna] Using {cv_folds}-fold cross validation")

    def objective(trial: optuna.trial.Trial):
        # Search space
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_sizes = []
        for i in range(num_layers):
            hidden_sizes.append(trial.suggest_categorical(f"n_units_l{i}", [16, 32, 48, 64, 96, 128, 256]))

        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "gelu"]) 

        batch_norm = trial.suggest_categorical("batch_norm", [False, True])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        scheduler_name = trial.suggest_categorical("scheduler", ["none", "step", "cosine"])
        if scheduler_name == "step":
            step_size = trial.suggest_int("step_size", 1, 5)
            gamma = trial.suggest_float("gamma", 0.1, 0.9)
        elif scheduler_name == "cosine":
            t_max = trial.suggest_int("t_max", 5, 50)

        patience = trial.suggest_int("patience", 1, 10)

        print(
            f"[optuna] Trial {trial.number}: "
            f"batch_size={batch_size}, num_layers={num_layers}, hidden_sizes={hidden_sizes}, "
            f"activation={activation}, batch_norm={batch_norm}, dropout={dropout:.3f}, lr={lr:.6g}, "
            f"scheduler={scheduler_name}, patience={patience}"
        )
        if scheduler_name == "step":
            print(f"[optuna] Trial {trial.number}: step_size={step_size}, gamma={gamma:.3f}")
        elif scheduler_name == "cosine":
            print(f"[optuna] Trial {trial.number}: t_max={t_max}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.MSELoss()

        fold_best_losses = []

        for fold_index, (train_idx, val_idx) in enumerate(kfold.split(x_array), start=1):
            print(
                f"[optuna] Trial {trial.number}: fold {fold_index}/{cv_folds} "
                f"train_samples={len(train_idx)}, val_samples={len(val_idx)}"
            )

            try:
                train_loader, val_loader = build_fold_loaders(
                    x_array,
                    y_array,
                    train_idx,
                    val_idx,
                    batch_size,
                    batch_norm,
                )
            except ValueError as exc:
                print(f"[optuna] Trial {trial.number}: fold {fold_index} invalid - {exc}")
                raise TrialPruned() from exc

            model = build_model(input_size, hidden_sizes, batch_norm, dropout, activation_name=activation, logBER=True).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            if scheduler_name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            else:
                scheduler = None

            # Early stopping variables
            best_val = float("inf")
            epochs_no_improve = 0

            # Training loop
            for epoch in range(n_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device).float().unsqueeze(1)

                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device).float()
                        yb = yb.to(device).float().unsqueeze(1)
                        preds = model(xb)
                        val_losses.append(criterion(preds, yb).item())

                val_loss = float(np.mean(val_losses))

                print(
                    f"[optuna] Trial {trial.number}: fold {fold_index}/{cv_folds}, "
                    f"epoch {epoch + 1}/{n_epochs}, val_loss={val_loss:.6f}, best_val={best_val:.6f}"
                )

                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(
                        f"[optuna] Trial {trial.number}: fold {fold_index} early stopping at epoch {epoch + 1} "
                        f"after {epochs_no_improve} non-improving epochs"
                    )
                    break

            fold_best_losses.append(best_val)
            cv_mean_loss = float(np.mean(fold_best_losses))
            print(
                f"[optuna] Trial {trial.number}: completed fold {fold_index}/{cv_folds}, "
                f"fold_best_val={best_val:.6f}, cv_mean_loss={cv_mean_loss:.6f}"
            )

            trial.report(cv_mean_loss, fold_index - 1)
            if trial.should_prune():
                print(f"[optuna] Trial {trial.number}: pruned after fold {fold_index} with cv_mean_loss={cv_mean_loss:.6f}")
                raise TrialPruned()

        final_cv_loss = float(np.mean(fold_best_losses))
        print(f"[optuna] Trial {trial.number}: completed with cv_loss={final_cv_loss:.6f}")
        return final_cv_loss

    # Create study
    if storage:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner(), storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner())

    print("[optuna] Beginning optimization...")
    study.optimize(objective, n_trials=n_trials)

    # Save best params
    out = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }

    out_path = os.path.join(os.getcwd(), "optuna_study_result.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study