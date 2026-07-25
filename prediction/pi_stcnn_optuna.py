import json
import os
import torch
import optuna
from prediction.l_freq_loss import l_freq_loss
from prediction.param_pred_optuna_helpers import *

def run_pi_stcnn_optuna(x_array, y_array, feature_columns, batch_size=16, mlp_hidden_map=None, tcnn_hidden_map=None, n_trials=20, n_epochs=550, seed=42,
                        study_name="pi_stcnn_optuna", storage=None, timeout_seconds=5.5*3600):

    set_seed(seed)

    if mlp_hidden_map is None:
        mlp_hidden_map = {
            "rect_long": [64, 64, 64],
            "funnel_short_large": [64, 128],
            "funnel_med_small": [32, 64, 128],
            "funnel_med_med": [64, 128, 256],
            "funnel_med_large": [128, 256, 512],
            "funnel_long_small": [32, 64, 128, 256],
            "funnel_long_large": [64, 128, 256, 512]
        }

        if tcnn_hidden_map is None:
            tcnn_hidden_map = {
                "rect_med": [64, 64, 64, 64, 64],
                "rect_large": [128, 128, 128, 128, 128],
                "rect_xl": [256, 256, 256, 256, 256]
            }

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")

    def objective(trial: optuna.trial.Trial):

        # Search Space
        mlp_hidden_shape_name = trial.suggest_categorical("mlp_hidden_shape_name", list(mlp_hidden_map.keys()))
        tcnn_hidden_shape_name = trial.suggest_categorical("tcnn_hidden_shape_name", list(tcnn_hidden_map.keys()))
        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.02)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-4, log=True)
        tcnn_1st_layer_kernel_size = trial.suggest_int("tcnn_1st_layer_kernel_size", 5, 16)
        M = trial.suggest_categorical("M", [1.5, 2.0])

        varience_min = 1.0

        mlp_hidden = mlp_hidden_map[mlp_hidden_shape_name]
        tcnn_out_channels = tcnn_hidden_map[tcnn_hidden_shape_name]
        tcnn_hidden = [
            [tcnn_out_channels[0], tcnn_1st_layer_kernel_size, 1],
            [tcnn_out_channels[1], 4, 2],
            [tcnn_out_channels[2], 4, 2],
            [tcnn_out_channels[3], 8, 2],
            [tcnn_out_channels[4], 4, 2],
        ]

        lr = 0.001
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = l_freq_loss()
        
        print(f"[optuna] Trial {trial.number}:")
        print(f"mlp_hidden:")
        print(mlp_hidden)
        print(f"tcnn_hidden:")
        print(tcnn_hidden)
        print(f"dropout: {dropout}")
        print(f"weight_decay: {weight_decay}")
        print(f"varience_min: {varience_min}")
        trial_loss = run_pi_stcnn_trial(trial, device, x_array, feature_columns, y_array, batch_size, mlp_hidden, tcnn_hidden,
                        dropout, M, varience_min, lr, weight_decay, n_epochs, criterion, seed)

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

    out_path = os.path.join(os.getcwd(), "optuna_param_study_result.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study 

