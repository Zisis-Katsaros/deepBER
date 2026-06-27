import torch
from prediction.param_pred_optuna import run_optuna


# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict_30perc.pt", weights_only=False)


x_array = pred_arrays_dict["param_prediction_test"][0]
s_dict = pred_arrays_dict["param_prediction_test"][1]
feature_columns = pred_arrays_dict["param_prediction_test"][6]


hidden_map = {
    "funnel_small": [64, 32, 32],
    "funnel_large": [128, 64, 48, 32],
    "rect_small": [64, 64, 64],
    "rect_large": [96, 96, 96, 96],
    "pyramid_small": [48, 64, 64, 48],
    "pyramid_large": [48, 64, 128, 64, 48]
}

storage_url = "sqlite:///single_mlp_optuna.db"
run_optuna("single_mlp", x_array, s_dict, feature_columns, batch_size=64, hidden_map=hidden_map, n_trials=90, n_epochs=25, seed=42, 
            study_name="single_mlp_optuna", storage=storage_url)
