import torch
from torch import nn
from load_set import create_param_prediction_arrays, create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor
from complexNN import nn as cvnn
from prediction.test_predictor_config import test_predictor_configuration
from rmse import RMSELoss
from dataset_manipulation import mock_pki
from prediction.param_pred_optuna import run_optuna

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

csv_names = [
    ["s_param_dataset_batch1.csv", "s_param_dataset_batch2.csv", "s_param_dataset_batch3.csv", "s_param_dataset_batch4.csv"],
]

test_names = ["param_prediction_test"]

test_info_dict = create_param_prediction_arrays(csv_names, test_names, sampling_method="lhs", subfolder="s_params")


x_array = test_info_dict["param_prediction_test"][0]
s_dict = test_info_dict["param_prediction_test"][1]
feature_columns = test_info_dict["param_prediction_test"][6]

# x_array, feature_columns = mock_pki(x_array, feature_columns, s12real)

elements = ["S55", "S78", "S217"]
for element in elements:
    for part in ["real", "imag"]:
        y_array = s_dict[element].real if part == "real" else s_dict[element].imag

        dataloader = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=128,
                        seed=42,
                        standard_scale=True,
                        split_method="lhs"
                        )
        
        predictor = DeepBER_Param_Predictor(
            input_size=len(feature_columns) - 1, 
            hidden=[128, 128, 128, 128], 
            activation_fn=nn.GELU(), 
            dropout=0.04
            )
        
        criterion = RMSELoss()
        learning_rate = 0.000446
        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
        
        test_predictor_configuration(
        title="{element}{part}",
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

# storage_url = "sqlite:///param_pred_study_v2.db"
# run_optuna(x_array, s_dict, feature_columns, selected_elements=None,n_trials=90, n_epochs=25, seed=42, 
#            study_name="param_pred_study_v2", storage=storage_url)


