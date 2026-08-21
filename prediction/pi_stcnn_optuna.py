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
            "rect_short_small": [64, 64, 64],
            "rect_long_small": [64, 64, 64, 64],
            "rect_short_large": [128, 128, 128],
            "rect_long_large": [128, 128, 128, 128],
            "rect_long_xl": [256, 256, 256, 256],
            "rect_long_xxl": [512, 512, 512, 512],
            "rect_long_xxxl": [1024, 1024, 1024, 1024],

            "funnel_xshort_small": [64, 128],
            "funnel_short_small": [32, 64, 128],
            "funnel_short_med": [64, 128, 256],
            "funnel_short_large": [128, 256, 512],
            "funnel_short_xl": [256, 512, 1024],
            "funnel_long_small": [32, 64, 128, 256],
            "funnel_long_large": [64, 128, 256, 512]
        }

        
        if tcnn_hidden_map is None:
            tcnn_hidden_map = {
                # "rect_3_128": [128, 128, 128],
                # "rect_3_196": [196, 196, 196],
                # "rect_3_256": [256, 256, 256],
                # "rect_3_384": [384, 384, 384],

                # "rect_4_128": [128, 128, 128, 128],
                # "rect_4_196": [196, 196, 196, 196],
                # "rect_4_256": [256, 256, 256, 256],
                # "rect_4_384": [384, 384, 384, 384],
                # "rect_4_512": [512, 512, 512, 512],
                # "rect_4_768": [768, 768, 768, 768],

                "rect_5_128": [128, 128, 128, 128, 128],
                "rect_5_196": [196, 196, 196, 196, 196],
                "rect_5_256": [256, 256, 256, 256, 256],
                "rect_5_384": [384, 384, 384, 384, 384],
                "rect_5_512": [512, 512, 512, 512, 512],
                "rect_5_768": [768, 768, 768, 768, 768],

                "rect_6_128": [128, 128, 128, 128, 128, 128],
                "rect_6_196": [196, 196, 196, 196, 196, 196],
                "rect_6_256": [256, 256, 256, 256, 256, 256],
                "rect_6_384": [384, 384, 384, 384, 384, 384],
                "rect_6_512": [512, 512, 512, 512, 512, 512],
                "rect_6_768": [768, 768, 768, 768, 768, 768],

                "rect_7_128": [128, 128, 128, 128, 128, 128, 128],
                "rect_7_196": [196, 196, 196, 196, 196, 196, 196],
                "rect_7_256": [256, 256, 256, 256, 256, 256, 256],
                "rect_7_384": [384, 384, 384, 384, 384, 384, 384],
                "rect_7_512": [512, 512, 512, 512, 512, 512, 512],
                "rect_7_768": [768, 768, 768, 768, 768, 768, 768],

                # "inv_funnel_4_small_narrow": [384, 256, 196, 128],
                # "inv_funnel_4_small_wide": [512, 384, 256, 128],
                # "inv_funnel_4_med_narrow": [512, 384, 256, 196],
                # "inv_funnel_4_med_wide": [768, 512, 384, 196],
                # "inv_funnel_4_large_narrow": [768, 512, 384, 256],
                # "inv_funnel_4_xl_narrow": [768, 512, 384, 384],

                "inv_funnel_5_small_narrow": [512, 384, 256, 196, 128],
                "inv_funnel_5_small_wide": [768, 512, 384, 256, 128],
                "inv_funnel_5_med_narrow": [768, 512, 384, 256, 196],
                "inv_funnel_5_large_narrow":[768, 512, 384, 256, 256],

                "inv_funnel_6_small": [768, 512, 384, 256, 196, 128],
                "inv_funnel_6_med": [768, 512, 384, 256, 196, 196],
                "inv_funnel_6_large": [768, 512, 384, 256, 256, 196],
                "inv_funnel_6_xl": [768, 512, 384, 384, 256, 256],
                }

    print(f"[optuna] Starting study '{study_name}' with n_trials={n_trials}, n_epochs={n_epochs}, seed={seed}")

    def objective(trial: optuna.trial.Trial):

        # Search Space:
        # MLP hyperparameters
        mlp_hidden_shape_name = trial.suggest_categorical("mlp_hidden_shape_name", list(mlp_hidden_map.keys()))
        mlp_hidden = mlp_hidden_map[mlp_hidden_shape_name]
        dropout = 0.0 # trial.suggest_float("dropout", 0.0, 0.2, step=0.02)

        

        # TCNN hyperparameters
        tcnn_hidden_shape_name = trial.suggest_categorical("tcnn_hidden_shape_name", list(tcnn_hidden_map.keys()))
        tcnn_hidden_shape = tcnn_hidden_map[tcnn_hidden_shape_name]
        tcnn_n_layers = len(tcnn_hidden_shape)

        tcnn_hidden = []
        tcnn_hidden.append([tcnn_hidden_shape[0], 0, 0])

        current_seq_len = 10 # Corresponds to self.initial_seq_len
        padding = 1

        N = y_array.shape[2]
        M = 1.5 # trial.suggest_categorical("M", [1.5, 2.0])
        target_len = int(N * M + 1)
        for i in range(1, tcnn_n_layers):
            out_channels = tcnn_hidden_shape[i]

            if i==0:
                stride = 1
                kernel_size = trial.suggest_int(f"tcnn_l{i}_kernel_size", 8, 32, step=4)
            elif i == tcnn_n_layers - 1:
                stride = trial.suggest_int(f"tcnn_l{i}_stride", 1, 3)
                if stride == 1:
                    kernel_size = trial.suggest_categorical(f"tcnn_l{i}_kernel_size_s1", [2, 3, 4])
                else:
                    multiplier = trial.suggest_int(f"tcnn_l{i}_c", 1, 2)
                    kernel_size = stride * multiplier
            else:
                stride = trial.suggest_int(f"tcnn_l{i}_stride", 2, 4)
                multiplier = trial.suggest_int(f"tcnn_l{i}_c", 1, 3)
                kernel_size = stride * multiplier

            # L_out = (L_in - 1) * stride - 2 * padding + kernel_size to calculate output length
            current_seq_len = (current_seq_len - 1) * stride - 2 * padding + kernel_size
            tcnn_hidden.append([out_channels, kernel_size, stride])

        # Prune if network generates fewer points than needed
        if current_seq_len < target_len:
            raise optuna.TrialPruned(f"Trial {trial.number} pruned due to insufficient output length: {current_seq_len} < {target_len}")

        if current_seq_len > target_len + 600:
            raise optuna.TrialPruned(f"Trial {trial.number} pruned due to excessive output length: {current_seq_len} > {target_len}")
        
        # Other hyperparameters
        weight_decay = 0.0 # trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        varience_min = 1.0
        lr = 0.001
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = l_freq_loss()

        # Print trial parameters
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
        reduction_factor=3
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

