import json
import os
import random
import numpy as np
import torch
import optuna
from optuna.exceptions import TrialPruned

from prediction.predictor import DeepBERPredictor
from load_set import load_csv_dataset, create_dataloader
from dataset_manipulation import extend_features, exclude_columns


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(input_size, hidden_sizes, batch_norm, dropout, logBER=True):
    return DeepBERPredictor(
        input_size=input_size,
        hidden=hidden_sizes,
        activation_fn=torch.nn.ReLU(),
        logBER=logBER,
        batch_norm=batch_norm,
        dropout=dropout,
    )


def run_optuna(csv_paths, n_trials=20, n_epochs=5, seed=42, study_name="deepber_optuna", storage=None):
    set_seed(seed)

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")

    x_array, y_array, feature_columns = load_csv_dataset(csv_paths, target_column="snr")

    print(f"[optuna] Loaded dataset: x_shape={x_array.shape}, y_shape={y_array.shape}, features={len(feature_columns)}")

    # Extra features (from main.py)
    # Width to space ratio:
    x_array, feature_columns = extend_features(x_array, feature_columns, "width", "space", "/", "width_space_ratio")
    
    # Cross-sectional area:
    x_array, feature_columns = extend_features(x_array, feature_columns, "width", "metal_thickness", "*", "cross_sectional_area")
    
    # Ground width to signal width ratio:
    x_array, feature_columns = extend_features(x_array, feature_columns, "gnd_width", "width", "/", "gnd_width_width_ratio")

    # Remove columns
    x_array, feature_columns = exclude_columns(x_array, feature_columns, columns_to_exclude=["delay"])

    print(f"[optuna] After feature engineering: x_shape={x_array.shape}, features={len(feature_columns)}")

    input_size = len(feature_columns)

    def objective(trial: optuna.trial.Trial):
        # Search space
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_sizes = []
        for i in range(num_layers):
            hidden_sizes.append(trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64, 128, 256]))

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
            f"batch_norm={batch_norm}, dropout={dropout:.3f}, lr={lr:.6g}, "
            f"scheduler={scheduler_name}, patience={patience}"
        )
        if scheduler_name == "step":
            print(f"[optuna] Trial {trial.number}: step_size={step_size}, gamma={gamma:.3f}")
        elif scheduler_name == "cosine":
            print(f"[optuna] Trial {trial.number}: t_max={t_max}")

        # Build dataloaders with drop_last=True to avoid BatchNorm issues with size-1 batches
        dataloaders = create_dataloader(x_array, y_array, batch_size=batch_size, ber_interval=[9.5, 14.5], seed=seed, standard_scale=True)
        train_loader, val_loader, _ = dataloaders
        
        # Recreate train_loader with drop_last=True
        train_dataset = train_loader.dataset
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_model(input_size, hidden_sizes, batch_norm, dropout, logBER=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        else:
            scheduler = None

        criterion = torch.nn.MSELoss()

        best_val = float("inf")
        epochs_no_improve = 0

        print(f"[optuna] Trial {trial.number}: training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")

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

            # Validation
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
                f"[optuna] Trial {trial.number}: epoch {epoch + 1}/{n_epochs}, "
                f"val_loss={val_loss:.6f}, best_val={best_val:.6f}"
            )

            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"[optuna] Trial {trial.number}: pruned at epoch {epoch + 1} with val_loss={val_loss:.6f}")
                raise TrialPruned()

            # Early stopping
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

        print(f"[optuna] Trial {trial.number}: completed with best_val={best_val:.6f}")
        return best_val

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

"""
if __name__ == "__main__":
    # Small smoke test when run directly
    csv_names = [
        "delay_snr_csv_database1.csv",
        "delay_snr_csv_database2.csv",
    ]
    csv_paths = [os.path.join(os.getcwd(), "csv_files", n) for n in csv_names]
    run_optuna(csv_paths, n_trials=20, n_epochs=100, seed=42)
"""