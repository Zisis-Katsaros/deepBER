import json
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
import optuna
from rmse import RMSELoss
from dataset_manipulation import extend_features
from prediction.param_pred_optuna_helpers import *

def run_optuna(model_architecture, x_array, s_dict, feature_columns, selected_elements=None, n_trials=20, n_epochs=5, seed=42, 
               study_name="param_pred_optuna", storage=None, timeout_seconds = 5.5 * 3600):
    set_seed(seed)

    if selected_elements is None:
        selected_elements = ["S55", "S78", "S217"]

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")
   
    groups = x_array[:, 0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(x_array, groups=groups))

    def objective(trial: optuna.trial.Trial):
        batch_size = 128

        use_width_space_ratio = trial.suggest_categorical("use_width_space_ratio", [True, False])

        x_xtnd = x_array[:, 1:].astype(np.float32)
        feat_cols_xtnd = feature_columns[1:].copy()
        if use_width_space_ratio:
            x_xtnd, feat_cols_xtnd = extend_features(x_xtnd, feat_cols_xtnd, "width", "space", "/", "width_space_ratio")
        
        x_xtnd, feat_cols_xtnd = extend_features(x_xtnd, feat_cols_xtnd, "width", "metal_thickness", "*", "cross_sectional_area")
        x_xtnd, feat_cols_xtnd = extend_features(x_xtnd, feat_cols_xtnd, "gnd_width", "width", "/", "gnd_width_width_ratio")

        hidden_map = {
            "funnel_steep": [256, 128, 64, 32],
            "funnel_shallow": [256, 128, 128, 64, 64],
            "funnel_long": [256, 256, 128, 128, 64, 64],
            "funnel_long_small": [256, 128, 64, 64, 32, 32],
            "rect_medium": [128, 128, 128, 128],
            "rect_large": [256, 256, 256, 256],
            "pyramid_short": [64, 128, 64, 32],
            "pyramid_small": [32, 64, 128, 64, 32],
            "pyramid_large": [64, 128, 256, 128, 64],
            "pyramid_large_short": [128, 256, 128, 64],
            "pyramid_large_long": [64, 128, 256, 128, 64, 32],
        }

        hidden_shape_name = trial.suggest_categorical("hidden_shape", list(hidden_map.keys()))
        
        hidden_sizes = hidden_map[hidden_shape_name]
        num_layers = len(hidden_sizes)

        activation = "gelu" 
        batch_norm = False

        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.02)
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        scheduler_name = "none"
   
        # scheduler_name = trial.suggest_categorical("scheduler", ["none", "step", "cosine"])
        step_size, gamma, t_max = None, None, None # initialized as none so that they can be passed in run_trial
        if scheduler_name == "step":
            step_size = trial.suggest_int("step_size", 5, 15)
            gamma = trial.suggest_float("gamma", 0.2, 0.8)
        elif scheduler_name == "cosine":
            t_max = trial.suggest_int("t_max", n_epochs//2, n_epochs)
        

        patience = 8

        print(
            f"[optuna] Trial {trial.number}: "
            f"batch_size={batch_size}, num_layers={num_layers}, hidden_sizes={hidden_sizes}, "
            f"activation={activation}, batch_norm={batch_norm}, dropout={dropout:.3f}, lr={lr:.6g}, "
            f"scheduler={scheduler_name}, patience={patience}"
        )
        print(f"use_width_space_ratio=True") if use_width_space_ratio else print(f"use_width_space_ratio=False")
        
        if scheduler_name == "step":
            print(f"[optuna] Trial {trial.number}: step_size={step_size}, gamma={gamma:.3f}")
        elif scheduler_name == "cosine":
            print(f"[optuna] Trial {trial.number}: t_max={t_max}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = RMSELoss()
        
        print(
            f"[optuna] Trial {trial.number}: "
            f"train_samples={len(train_idx)}, val_samples={len(val_idx)}"
        )

        current_losses  = run_trial(trial, device, model_architecture, selected_elements, x_xtnd, feat_cols_xtnd, s_dict, train_idx, val_idx, batch_size, 
            batch_norm, hidden_sizes, dropout, activation, lr, scheduler_name, step_size, gamma, t_max, n_epochs, criterion, patience)



        avg_loss = float(np.mean(current_losses))
        print(f"[optuna] Trial {trial.number}: completed with avg_loss={avg_loss:.6f}")
        return avg_loss

    if storage:
        study = optuna.create_study(direction="minimize", study_name=study_name, 
                                    sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True), 
                                    pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,  
                                            max_resource=len(selected_elements)*2, 
                                            reduction_factor=3 
                                        ), storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name, 
                                    sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True), 
                                    pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,  
                                            max_resource=len(selected_elements)*2,
                                            reduction_factor=3 
                                        ))

    print("[optuna] Beginning optimization...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

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