import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.param_pred_optuna import run_optuna
from load_set import create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor_Complex
from rmse import RMSELoss
from prediction.test_predictor_config import test_predictor_configuration, single_geometry_test
from dataset_manipulation import pki_extend
from complexNN import nn as cvnn
import numpy as np
from export_files_for_transient import export_files_for_transient


# ============================================= Initializing Dataset ============================================= #
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict_total.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

"""
hidden_map = {
    "funnel_small": [96, 64, 48, 32],
    "funnel_large": [128, 96, 64, 48, 32],
    "pyramid_small": [48, 64, 64, 48],
    "pyramid_large": [48, 64, 96, 64, 48]
}


storage_url = "sqlite:///cv_mlp_optuna_v2.db"
run_optuna("cv_mlp", x_array, s_dict, feature_columns, batch_size=128, hidden_map=hidden_map, n_trials=50, n_epochs=25, seed=42, 
            study_name="cv_mlp_optuna_v2", storage=storage_url)
"""

geoms_tested = 44
labels_dict_per_geom = np.array([{} for _ in range(geoms_tested)], dtype=dict)
preds_dict_per_geom = np.array([{} for _ in range(geoms_tested)], dtype=dict)
freq_arrays_per_geom = [None for _ in range(geoms_tested)]

processed_elements = [key for idx, key in enumerate(s_dict.keys()) if idx < -1 and key != "all"] 
elements = list(key for key in s_dict.keys() if key != "all")
for element in elements:
    if element in processed_elements:
        continue
    print(f"Training and testing for {element}\n")

    y_array = s_dict[element]
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params, *_ = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    predictor = DeepBER_Param_Predictor_Complex(
        input_size=len(feature_columns), 
        hidden=[128, 96, 64, 48], 
        activation_fn=cvnn.cGelu(), 
        dropout=0.02
        ).to(device)
    
    criterion = RMSELoss()
    learning_rate = 9.906773918756151e-05
    weight_decay = 0.0007214881131497281
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    test_predictor_configuration(
    title=f"{element}",
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
    test_out_dir = f"out_files/cv_mlp"
    )
    
for element in elements:
    y_array = s_dict[element]
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params, *_ = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    predictor = DeepBER_Param_Predictor_Complex(
            input_size=len(feature_columns), 
            hidden=[128, 96, 64, 48], 
            activation_fn=cvnn.cGelu(), 
            dropout=0.02
            ).to(device)
    predictor.load_state_dict(torch.load(f"out_files/cv_mlp/weights/best_model_{element}.pth", map_location=device))
# """

    geometries, labels_per_geom, preds_per_geom, freq_arrays = single_geometry_test(
        title=f"{element}",
        device=device,
        model=predictor,
        test_data = dataloader[2],
        x_scale_params=x_scale_params,
        y_scale_params=y_scale_params,
        max_geoms=geoms_tested,
        visualization=False,
        max_figures=46,
        save_dir = f"out_files/cv_mlp/pred_vs_act_plots/{element}",
        close_figures=True
    )

    for geom_idx, (label_array, pred_array, geom_freq_array) in enumerate(zip(labels_per_geom, preds_per_geom, freq_arrays)):
        labels_dict_per_geom[geom_idx][element] = label_array
        preds_dict_per_geom[geom_idx][element] = pred_array
        if freq_arrays_per_geom[geom_idx] is None:
            freq_arrays_per_geom[geom_idx] = geom_freq_array

feature_columns_no_freq = [col for col in feature_columns if col != "frequency_ghz"]
export_files_for_transient(
    geometries=np.array(geometries), 
    feature_names=feature_columns_no_freq, 
    labels_dict_per_geom=labels_dict_per_geom, 
    preds_dict_per_geom=preds_dict_per_geom, 
    freq_arrays_per_geom=freq_arrays_per_geom,
    save_dir=f"out_files/cv_mlp/touchstone"
    )