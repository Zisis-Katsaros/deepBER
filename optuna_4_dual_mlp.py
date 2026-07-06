import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.param_pred_optuna import run_optuna
from load_set import create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor
from rmse import RMSELoss
from prediction.test_predictor_config import test_predictor_configuration, single_geometry_test, abcd_preds_vs_act_freq
import numpy as np
from dataset_manipulation import pki_extend


# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

pki = False
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
s_mock_dict = torch.load("csv_files/s_params/pt/s_mock_dict.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

"""
hidden_map = {
    "funnel_small": [128, 96, 64, 48],
    "funnel_large": [256, 128, 96, 64, 48],
    "pyramid_small": [64, 96, 128, 96, 64],
    "pyramid_large": [64, 128, 256, 128, 64]
}

storage_url = "sqlite:///dual_mlp_optuna_v2.db"
run_optuna("dual_mlp", x_array, s_dict, feature_columns, batch_size=128, hidden_map=hidden_map, n_trials=50, n_epochs=25, seed=42, 
            study_name="dual_mlp_optuna_v2", storage=storage_url)
"""

geoms_tested = 5
labels_dict_per_geom = np.array([{} for _ in range(geoms_tested)], dtype=dict)
preds_dict_per_geom = np.array([{} for _ in range(geoms_tested)], dtype=dict)
freq_arrays_per_geom = [None for _ in range(geoms_tested)]

processed_elements = [key for idx, key in enumerate(s_dict.keys()) if idx < 200 and key != "all"] 
elements = list(key for key in s_dict.keys() if key != "all") # ["S55", "S78", "S217"]
for element in elements:
    if element in processed_elements:
        continue
    print(f"Training and testing for {element}\n")
    if pki:
        x_xtnd, feature_columns_xtnd = pki_extend(x_array, feature_columns, s_mock_dict[element])
    else:
        x_xtnd = x_array
    y_array = np.stack([s_dict[element].real, s_dict[element].imag], axis=1)
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params = create_param_dataloader(
                    x_xtnd,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    predictor = DeepBER_Param_Predictor(
        input_size=len(feature_columns), 
        hidden=[48, 64, 96, 64, 48], 
        activation_fn=nn.GELU(), 
        output_size=out_size,
        dropout=0.02
        ).to(device)
    
    criterion = RMSELoss()
    learning_rate = 9.291819656627535e-05
    weight_decay = 5.976118759714283e-06
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
    test_out_dir = f"out_files/dual_mlp/{element}/pki" if pki else f"out_files/dual_mlp/{element}/no_pki",
    close_figures=True
    )

# """
for element in elements:
    if pki:
        x_xtnd, feature_columns_xtnd = pki_extend(x_array, feature_columns, s_mock_dict[element])
    else:
        x_xtnd = x_array
    y_array = np.stack([s_dict[element].real, s_dict[element].imag], axis=1)
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params = create_param_dataloader(
                    x_xtnd,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    predictor = DeepBER_Param_Predictor(
        input_size=len(feature_columns), 
        hidden=[48, 64, 96, 64, 48], 
        activation_fn=nn.GELU(), 
        output_size=out_size,
        dropout=0.02
        ).to(device) 
    predictor.load_state_dict(torch.load(f"out_files/dual_mlp/{element}/pki/best_model.pth" if pki else f"out_files/dual_mlp/{element}/no_pki/best_model.pth", map_location=device))
# """
    
    labels_per_geom, preds_per_geom, freq_arrays = single_geometry_test(
        title=f"{element}",
        device=device,
        model=predictor,
        test_data = dataloader[2],
        x_scale_params=x_scale_params,
        y_scale_params=y_scale_params,
        max_geoms=geoms_tested,
        pki=pki,
        save_dir = f"out_files/dual_mlp/{element}/pki" if pki else f"out_files/dual_mlp/{element}/no_pki",
        close_figures=True
    )

    for geom_idx, (label_array, pred_array, geom_freq_array) in enumerate(zip(labels_per_geom, preds_per_geom, freq_arrays)):
        labels_dict_per_geom[geom_idx][element] = label_array[:, 0] + 1j * label_array[:, 1]
        preds_dict_per_geom[geom_idx][element] = pred_array[:, 0] + 1j * pred_array[:, 1]
        if freq_arrays_per_geom[geom_idx] is None:
            freq_arrays_per_geom[geom_idx] = geom_freq_array

for geom_idx, (s_labels_dict, s_preds_dict, geom_freq_array) in enumerate(zip(labels_dict_per_geom, preds_dict_per_geom, freq_arrays_per_geom), start=1):
    if not s_labels_dict or geom_freq_array is None:
        continue
    abcd_preds_vs_act_freq(
        s_labels_dict,
        s_preds_dict,
        geom_freq_array,
        save_dir=f"out_files/dual_mlp/abcd_preds_vs_act/geom_{geom_idx}",
        close_figures=True
    )

