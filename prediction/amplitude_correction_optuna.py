import json
import os
import torch
import optuna
from rmse import RMSELoss
from prediction.param_pred_optuna_helpers import *

def run_amp_corr_optuna(x_array, y_array, feature_columns, batch_size=16, hidden_map=None, n_trials=20, n_epochs=100, seed=42,
                        study_name="amplitude_correction_optuna", storage=None, timeout_seconds=5.5*3600):
    set_seed(seed)

    if hidden_map is None:
        hidden_map = {
            "rect_small_short": [64, 64, 64],
            "rect_small_long": [64, 64, 64, 64],
            "rect_large_short": [128, 128, 128],
            "rect_large_long": [128, 128, 128, 128],
            "pyramid_small": [48, 64, 128, 64, 48],
            "pyramid_large": [64, 128, 256, 128, 64],
        }

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")

    def objective(trial: optuna.trial.Trial):
        # Search Space
        hidden_shape_name = trial.suggest_categorical("hidden_shape_name", list(hidden_map.keys()))
        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.02)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

        lr = 0.001
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = RMSELoss()

        hidden = hidden_map[hidden_shape_name]

        print(f"[optuna] Trial {trial.number}:")
        print(f"hidden:")
        print(hidden)
        print(f"dropout: {dropout}")
        print(f"weight_decay: {weight_decay}")

        trial_loss = run_amp_corr_trial(trial, device, x_array, feature_columns, y_array, batch_size, hidden, dropout, lr, weight_decay, n_epochs, criterion, seed)

        print(f"[optuna] Trial {trial.number}: completed with loss={trial_loss:.6f}")
        return trial_loss

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=100,
        reduction_factor=2
        )

    if storage:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler, pruner=pruner, storage=storage, load_if_exists=True)
        
        completed_trials = len(study.trials)
        trials_remaining = n_trials - completed_trials

        if trials_remaining <= 0:
            print(f"[optuna] Study '{study_name}' already has {completed_trials} completed trials, which meets or exceeds the requested {n_trials} trials. No further optimization will be performed.")
            return study
        print("[optuna] Beginning optimization...")
        study.optimize(objective, n_trials=trials_remaining, timeout=timeout_seconds)

    else:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler, pruner=pruner)

        print("[optuna] Beginning optimization...")
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    out = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }

    out_path = os.path.join(os.getcwd(), "optuna_amp_corr_study_result.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study