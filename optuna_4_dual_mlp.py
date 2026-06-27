import torch
from prediction.param_pred_optuna import run_optuna


# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)


x_array = pred_arrays_dict["param_prediction_test"][0]
s_dict = pred_arrays_dict["param_prediction_test"][1]
feature_columns = pred_arrays_dict["param_prediction_test"][6]


hidden_map = {
    "funnel_small": [128, 96, 64, 48],
    "funnel_large": [256, 128, 96, 64, 48],
    "pyramid_small": [64, 96, 128, 96, 64],
    "pyramid_large": [64, 128, 256, 128, 64]
}

storage_url = "sqlite:///dual_mlp_optuna_v2.db"
run_optuna("dual_mlp", x_array, s_dict, feature_columns, batch_size=128, hidden_map=hidden_map, n_trials=50, n_epochs=25, seed=42, 
            study_name="dual_mlp_optuna_v2", storage=storage_url)
