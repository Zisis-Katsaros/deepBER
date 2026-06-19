import json
import os
import random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
import optuna
from optuna.exceptions import TrialPruned
from rmse import RMSELoss
from torch.utils.data import DataLoader, TensorDataset
from prediction.predictor import DeepBER_Param_Predictor
from dataset_manipulation import extend_features

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_fold_loaders(x_array, y_array, train_idx, val_idx, batch_size, batch_norm):
    x_train = x_array[train_idx].astype(np.float32)
    y_train = y_array[train_idx].astype(np.float32)
    x_val = x_array[val_idx].astype(np.float32)
    y_val = y_array[val_idx].astype(np.float32)

    # X standardization
    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)
    train_std = np.where(train_std == 0.0, 1.0, train_std)

    x_train = ((x_train - train_mean) / train_std).astype(np.float32)
    x_val = ((x_val - train_mean) / train_std).astype(np.float32)

    # Y standardization
    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)
    y_train_std = np.where(y_train_std == 0.0, 1.0, y_train_std)

    y_train = (y_train - y_train_mean) / y_train_std
    y_val = (y_val - y_train_mean) / y_train_std

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

def build_model(input_size, hidden_sizes, batch_norm, dropout, activation_name="relu"):
    # Map activation name to a PyTorch activation module
    activation_map = {
        "relu": torch.nn.ReLU(),
        "leaky_relu": torch.nn.LeakyReLU(negative_slope=0.01),
        "elu": torch.nn.ELU(),
        "gelu": torch.nn.GELU(),
    }
    activation_fn = activation_map.get(activation_name, torch.nn.ReLU())

    return DeepBER_Param_Predictor(
        input_size=input_size,
        hidden=hidden_sizes,
        activation_fn=activation_fn,
        batch_norm=batch_norm,
        dropout=dropout,
    )


def run_optuna(x_array, s_dict, feature_columns, selected_elements=None,n_trials=20, n_epochs=5, seed=42, study_name="param_pred_optuna", 
               storage=None, cv_folds=3):
    set_seed(seed)

    if selected_elements is None:
        selected_elements = ["S22", "S55", "S34", "S78", "S217"]

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")
    # print(f"[optuna] Loaded dataset: x_shape={x_array.shape}, y_shape={y_array.shape}, features={len(feature_columns)}")

    input_size = len(feature_columns)
    # if cv_folds < 2:
    #     raise ValueError("cv_folds must be at least 2.")
    
    groups = x_array[:, 0]
    kfold = GroupKFold(n_splits=cv_folds)
    print(f"[optuna] Using {cv_folds}-fold cross validation")

    def objective(trial: optuna.trial.Trial):
        # Search space
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        data_manipulation = []
        if trial.suggest_categorical("use_width_space_ratio", [True, False]):
            data_manipulation.append("width_space_ratio")
        if trial.suggest_categorical("use_cross_sectional_area", [True, False]):
            data_manipulation.append("cross_sectional_area")
        if trial.suggest_categorical("use_gnd_width_width_ratio", [True, False]):
            data_manipulation.append("gnd_width_width_ratio")

        x_xtnd = x_array[:, 1:].astype(np.float32)
        feat_cols_xtnd = feature_columns[1:].copy()
        if "width_space_ratio" in data_manipulation:
            x_xtnd, feat_cols_xtnd = extend_features(x_xtnd, feat_cols_xtnd, "width", "space", "/", "width_space_ratio")
        if "cross_sectional_area" in data_manipulation:
            x_xtnd, feat_cols_xtnd = extend_features(x_xtnd, feat_cols_xtnd, "width", "metal_thickness", "*", "cross_sectional_area")
        if "gnd_width_width_ratio" in data_manipulation:
            x_xtnd, feat_cols_xtnd = extend_features(x_xtnd, feat_cols_xtnd, "gnd_width", "width", "/", "gnd_width_width_ratio")

        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_sizes = []
        for i in range(num_layers):
            hidden_sizes.append(trial.suggest_categorical(f"n_units_l{i}", [16, 32, 48, 64, 96, 128, 256]))

        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "gelu"]) 
        batch_norm = trial.suggest_categorical("batch_norm", [False, True])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)

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
        criterion = RMSELoss()
        
        cv_fold_scores = [] # Tracks score of each fold
        fold_best_losses = []


        for fold_index, (train_idx, val_idx) in enumerate(kfold.split(X=x_array, groups=groups)):
            print(
                f"[optuna] Trial {trial.number}: fold {fold_index}/{cv_folds} "
                f"train_samples={len(train_idx)}, val_samples={len(val_idx)}"
            )

            current_fold_losses = []

            for element in selected_elements:
                for part in ["Re", "Im"]:
                    y_array = s_dict[element].real if part == "Re" else s_dict[element].imag
                    print(f"[optuna] Training on {part}({element})")

                    try:
                        train_loader, val_loader = build_fold_loaders(
                            x_xtnd,
                            y_array,
                            train_idx,
                            val_idx,
                            batch_size,
                            batch_norm,
                        )
                    except ValueError as exc:
                        print(f"[optuna] Trial {trial.number}: fold {fold_index} invalid - {exc}")
                        raise TrialPruned() from exc
                    
                    model = build_model(len(feat_cols_xtnd), hidden_sizes, batch_norm, dropout, activation_name=activation).to(device)

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
                        
                current_fold_losses.append(best_val)

            fold_max_loss = float(np.max(current_fold_losses))
            cv_fold_scores.append(fold_max_loss) 
            print(
                f"[optuna] Trial {trial.number}: completed fold {fold_index}/{cv_folds}, "
                f"fold_best_val={best_val:.6f}, fold_max_loss={fold_max_loss:.6f}"
            )

            trial.report(fold_max_loss, fold_index)
            if trial.should_prune():
                print(f"[optuna] Trial {trial.number}: pruned after fold {fold_index}")
                raise TrialPruned()

        final_cv_loss = float(np.mean(cv_fold_scores))
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

    out_path = os.path.join(os.getcwd(), "optuna_param_study_result.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study