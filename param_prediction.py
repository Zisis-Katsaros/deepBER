import torch
from torch import nn
from load_set import create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor, DeepBER_Param_Predictor_Complex
from complexNN import nn as cvnn
from prediction.test_predictor_config import test_predictor_configuration
from rmse import RMSELoss
from dataset_manipulation import mock_pki
from prediction.param_pred_optuna import run_optuna
import numpy as np

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
pred_arrays_dict_10perc = torch.load("csv_files/s_params/pt/pred_arrays_dict_10perc.pt", weights_only=False)


x_array = pred_arrays_dict_10perc["param_prediction_test"][0]
s_dict = pred_arrays_dict_10perc["param_prediction_test"][1]
feature_columns = pred_arrays_dict_10perc["param_prediction_test"][6]

# x_array, feature_columns = mock_pki(x_array, feature_columns, s12real)
"""
elements = ["S55", "S78", "S217"]
for element in elements:
    # for part in ["real", "imag"]:
    for part in range(1):
        # y_array = s_dict[element].real if part == "real" else s_dict[element].imag
        y_array = np.stack([s_dict[element].real, s_dict[element].imag], axis=1)
        y_array = s_dict[element]
        out_size = y_array.shape[1] if y_array.ndim > 1 else 1

        dataloader = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=128,
                        seed=42,
                        standard_scale=True,
                        split_method="lhs"
                        )
        
        predictor = DeepBER_Param_Predictor_Complex(
            input_size=len(feature_columns) - 1, 
            hidden=[128, 128, 128, 128], 
            activation_fn=cvnn.cGelu(), 
            dropout=0.04
            ).to(device)
        
        criterion = RMSELoss()
        learning_rate = 0.000446
        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
        
        test_predictor_configuration(
        title=f"{element}{part}",
        device=device,
        model=predictor,
        dataloader=dataloader,
        learning_rate=learning_rate,
        batch_size=128,
        criterion=criterion,
        optimizer=optimizer,
        epochs=300,
        early_stopping=True,
        patience=20,
        training_curves=True,
        predicted_vs_actual=True,
        test_out_dir = f"out_files/{element}_{part}"
    )
"""

# storage_url = "sqlite:///param_pred_study_v2.db"
run_optuna("cv_mlp", x_array, s_dict, feature_columns, selected_elements=None,n_trials=90, n_epochs=25, seed=42, 
            study_name="param_pred_study")


