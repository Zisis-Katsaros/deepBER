import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.param_pred_optuna import run_optuna
from load_set import create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor
from rmse import RMSELoss
from prediction.test_predictor_config import test_predictor_configuration, single_geometry_test
from dataset_manipulation import pki_extend



# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

pki = False
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
s_mock_dict = torch.load("csv_files/s_params/pt/s_mock_dict.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"]
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

"""
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
"""

elements = ["S514"] #["S55", "S78", "S217"]
for element in elements:  
    if pki:
        x_xtnd_real, feature_columns_xtnd_real = pki_extend(x_array.copy(), feature_columns.copy(), s_mock_dict[element], mode="real")
        x_xtnd_imag, feature_columns_xtnd_imag = pki_extend(x_array.copy(), feature_columns.copy(), s_mock_dict[element], mode="imag")
    else:
        x_xtnd_real = x_array
        x_xtnd_imag = x_array
        feature_columns_xtnd_real = feature_columns
        feature_columns_xtnd_imag = feature_columns
    for part in ["real", "imag"]:
        
        y_array = s_dict[element].real if part == "real" else s_dict[element].imag
        out_size = y_array.shape[1] if y_array.ndim > 1 else 1

        dataloader, x_scale_params, y_scale_params = create_param_dataloader(
                        x_xtnd_real if part == "real" else x_xtnd_imag,
                        y_array,
                        batch_size=128,
                        seed=42,
                        standard_scale=True,
                        split_method="lhs"
                        )
        
        predictor = DeepBER_Param_Predictor(
            input_size=x_xtnd_real.shape[1] if part == "real" else x_xtnd_imag.shape[1], 
            hidden=[32, 48, 64, 48, 32], 
            activation_fn=nn.GELU(), 
            output_size=out_size,
            dropout=0.04
            ).to(device)
        
        criterion = RMSELoss()
        learning_rate = 0.000446
        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        test_predictor_configuration(
        title=f"{element}{part}",
        device=device,
        model=predictor,
        dataloader=dataloader,
        learning_rate=learning_rate,
        batch_size=128,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=300,
        early_stopping=True,
        patience=10,
        y_scale_params=y_scale_params,
        training_curves=True,
        predicted_vs_actual=True,
        test_out_dir = f"out_files/single_mlp/{element}/{part}/pki" if pki else f"out_files/single_mlp/{element}/{part}/no_pki"
        )
        
        single_geometry_test(
            title=f"{element}{part}",
            device=device,
            model=predictor,
            test_data = dataloader[2],
            x_scale_params=x_scale_params,
            y_scale_params=y_scale_params,
            max_geoms=5,
            pki=pki,
            save_dir = f"out_files/single_mlp/{element}/{part}/pki" if pki else f"out_files/single_mlp/{element}/{part}/no_pki"
        )