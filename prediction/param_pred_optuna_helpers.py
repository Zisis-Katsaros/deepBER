import random
import numpy as np
import torch
from torch import nn
from complexNN import nn as cvnn
from optuna.exceptions import TrialPruned
from torch.utils.data import DataLoader, TensorDataset
from prediction.predictor import DeepBER_Param_Predictor, DeepBER_Param_Predictor_Complex


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fold_loaders(x_array, y_array, train_idx, val_idx, batch_size, batch_norm):
    x_train = x_array[train_idx]
    y_train = y_array[train_idx]
    y_train = y_train.reshape(y_train.shape[0], -1)
    x_val = x_array[val_idx]
    y_val = y_array[val_idx]
    y_val = y_val.reshape(y_val.shape[0], -1)

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


def build_model(model_architecture, input_size, hidden_sizes, batch_norm, dropout, activation_name="relu"):
    # Map activation name to a PyTorch activation module
    activation_map = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "c_relu": cvnn.cRelu(),
        "c_leaky_relu": cvnn.cLeakyRelu(),
        "c_elu": cvnn.cElu(),
        "c_gelu": cvnn.cGelu()
    }
    if model_architecture == "cv_mlp":
        activation_name = "c_"+activation_name
    activation_fn = activation_map.get(activation_name, torch.nn.ReLU())

    if model_architecture == "single_mlp":
        return DeepBER_Param_Predictor(
            input_size=input_size,
            hidden=hidden_sizes,
            output_size=1,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            dropout=dropout,
        )
    elif model_architecture == "dual_mlp":
        return DeepBER_Param_Predictor(
            input_size=input_size,
            hidden=hidden_sizes,
            output_size=2,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            dropout=dropout,
        )
    elif model_architecture == "cv_mlp":
        return DeepBER_Param_Predictor_Complex(
            input_size=input_size,
            hidden=hidden_sizes,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            dropout=dropout,
        )
    else:
        raise ValueError("model_architecture should be either 'single_mlp', 'dual_mlp' or 'cv_mlp'.")


def run_trial(trial, device, model_architecture, selected_elements, x_pure, feat_cols_pure, s_dict, train_idx, val_idx, batch_size, 
              batch_norm, hidden_sizes, dropout, activation, lr, wd, scheduler_name, step_size, gamma, t_max, n_epochs, criterion, patience):
    current_losses = []
    step_idx = 0
    
    for element in selected_elements:
        if model_architecture == "single_mlp":
            for part in ["Re", "Im"]:
                y_array = s_dict[element].real if part == "Re" else s_dict[element].imag
                print(f"[optuna] Training on {part}({element})")

                try:
                    train_loader, val_loader = build_fold_loaders(
                        x_pure,
                        y_array,
                        train_idx,
                        val_idx,
                        batch_size,
                        batch_norm,
                    )
                except ValueError as exc:
                    print(f"[optuna] Trial {trial.number}: invalid - {exc}")
                    raise TrialPruned() from exc
                
                model = build_model(model_architecture, len(feat_cols_pure), hidden_sizes, batch_norm, dropout, activation_name=activation).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                if scheduler_name == "step":
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                elif scheduler_name == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
                else:
                    scheduler = None

                best_val = float("inf")
                epochs_no_improve = 0

                for epoch in range(n_epochs):
                    model.train()
                    for xb, yb in train_loader:
                        xb = xb.to(device).float()
                        yb = yb.to(device).float()

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
                            yb = yb.to(device).float()
                            preds = model(xb)
                            val_losses.append(criterion(preds, yb).item())

                    val_loss = float(np.mean(val_losses))

                    print(
                        f"[optuna] Trial {trial.number}: "
                        f"epoch {epoch + 1}/{n_epochs}, val_loss={val_loss:.6f}, best_val={best_val:.6f}"
                    )

                    if val_loss < best_val - 1e-8:
                        best_val = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        print(
                            f"[optuna] Trial {trial.number}: early stopping at epoch {epoch + 1} "
                            f"after {epochs_no_improve} non-improving epochs"
                        )
                        break
            
        else:
            if model_architecture == "dual_mlp":
                y_array = np.stack([s_dict[element].real, s_dict[element].imag], axis=1)
            elif model_architecture == "cv_mlp":
                y_array = s_dict[element]
            else:
                raise ValueError("model_architecture should be either 'single_mlp', 'dual_mlp' or 'cv_mlp'.")
            
            print(f"[optuna] Training on {element}")

            try:
                train_loader, val_loader = build_fold_loaders(
                    x_pure,
                    y_array,
                    train_idx,
                    val_idx,
                    batch_size,
                    batch_norm,
                )
            except ValueError as exc:
                print(f"[optuna] Trial {trial.number}: invalid - {exc}")
                raise TrialPruned() from exc
            
            model = build_model(model_architecture, len(feat_cols_pure), hidden_sizes, batch_norm, dropout, 
                                activation_name=activation).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            if scheduler_name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            else:
                scheduler = None

            best_val = float("inf")
            epochs_no_improve = 0

            for epoch in range(n_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

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
                        xb = xb.to(device)
                        yb = yb.to(device)
                        preds = model(xb)
                        val_losses.append(criterion(preds, yb).item())

                val_loss = float(np.mean(val_losses))

                print(
                    f"[optuna] Trial {trial.number}: "
                    f"epoch {epoch + 1}/{n_epochs}, val_loss={val_loss:.6f}, best_val={best_val:.6f}"
                )

                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(
                        f"[optuna] Trial {trial.number}: early stopping at epoch {epoch + 1} "
                        f"after {epochs_no_improve} non-improving epochs"
                    )
                    break    
            
        current_losses.append(best_val)

        trial.report(best_val, step_idx)
        if trial.should_prune():
            print(f"[optuna] Trial {trial.number}: pruned after step {step_idx}")
            raise TrialPruned()
        step_idx += 1
    return current_losses            
    