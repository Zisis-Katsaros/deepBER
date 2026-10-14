import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.param_pred_optuna import run_optuna
from load_set import create_param_dataloader, add_samples_for_extrapolation, organize_dataset_for_pi_stcnn
from prediction.predictor import DeepBER_Param_Predictor
from rmse import RMSELoss
from prediction.test_predictor_config import test_predictor_configuration, single_geometry_test
from dataset_manipulation import pki_extend
import numpy as np
from prediction.parameter_computations import s_param_imag_part_hilbert_construction
from export_files_for_transient import export_files_for_transient, convert_stcnn_outputs_to_dicts
from visualization import plot_preds_vs_act_freq

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict_total.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

geoms_tested = 44
labels_dict_per_geom = np.array([{} for _ in range(geoms_tested)], dtype=dict)
preds_dict_per_geom = np.array([{} for _ in range(geoms_tested)], dtype=dict)
freq_arrays_per_geom = [None for _ in range(geoms_tested)]

s_non_causal_dict = {}
first_unprocessed_element_idx = -1 # For continuing training on HPC server if 6hr time limit is reached on the first run. Set to -1 to start from scratch
processed_elements = [key for idx, key in enumerate(s_dict.keys()) if idx < first_unprocessed_element_idx and key != "all"]
elements = list(key for key in s_dict.keys() if key != "all") 
for element in elements:   
    if element in processed_elements:
        continue
    print(f"Training and testing for {element}\n")

    for part in ["Re", "Im"]:
        y_array = s_dict[element].real if part == "Re" else s_dict[element].imag
        out_size = y_array.shape[1] if y_array.ndim > 1 else 1

        dataloader, x_scale_params, y_scale_params, *_ = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=128,
                        seed=42,
                        standard_scale=True,
                        split_method="lhs"
                        )
    
        predictor = DeepBER_Param_Predictor(
            input_size=x_array.shape[1],
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
        title=f"{part}({element})",
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
        test_out_dir = f"out_files/single_mlp",
        close_figures=True
        )

# Forward
for element in elements:
    for part in ["Re", "Im"]:
        y_array = s_dict[element].real if part == "Re" else s_dict[element].imag
        out_size = y_array.shape[1] if y_array.ndim > 1 else 1

        dataloader, x_scale_params, y_scale_params, *_ = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=128,
                        seed=42,
                        standard_scale=True,
                        split_method="lhs"
                        )
    

        predictor = DeepBER_Param_Predictor(
            input_size=x_array.shape[1],
            hidden=[32, 48, 64, 48, 32],
            activation_fn=nn.GELU(),
            output_size=out_size,
            dropout=0.04
            ).to(device)
        predictor.load_state_dict(torch.load(f"out_files/single_mlp/weights/best_model_{part}({element}).pth", map_location=device))

        if part == "Re":
            geometries, labels_per_geom_real, preds_per_geom_real, freq_arrays = single_geometry_test(
                    title=f"{part}({element})",
                    device=device,
                    model=predictor,
                    test_data = dataloader[2],
                    x_scale_params=x_scale_params,
                    y_scale_params=y_scale_params,
                    max_geoms=geoms_tested,
                    visualization=True,
                    save_dir = f"out_files/single_mlp/pred_vs_act_plots/{part}({element})",
                    close_figures=True
                )
        else:
            geometries, labels_per_geom_imag, preds_per_geom_imag, freq_arrays = single_geometry_test(
                    title=f"{part}({element})",
                    device=device,
                    model=predictor,
                    test_data = dataloader[2],
                    x_scale_params=x_scale_params,
                    y_scale_params=y_scale_params,
                    max_geoms=geoms_tested,
                    save_dir = f"out_files/single_mlp/pred_vs_act_plots/{part}({element})",
                    close_figures=True
                )
    labels_per_geom = np.stack([labels_per_geom_real, labels_per_geom_imag], axis=2)
    preds_per_geom = np.stack([preds_per_geom_real, preds_per_geom_imag], axis=2)
 
    for geom_idx, (label_array, pred_array, geom_freq_array) in enumerate(zip(labels_per_geom, preds_per_geom, freq_arrays)):
            labels_dict_per_geom[geom_idx][element] = label_array[:, 0] + 1j * label_array[:, 1]
            preds_dict_per_geom[geom_idx][element] = pred_array[:, 0] + 1j * pred_array[:, 1]
            if freq_arrays_per_geom[geom_idx] is None:
                freq_arrays_per_geom[geom_idx] = geom_freq_array

feature_columns_no_freq = [col for col in feature_columns if col != "frequency_ghz"]
export_files_for_transient(
    geometries=np.array(geometries), 
    feature_names=feature_columns_no_freq, 
    labels_dict_per_geom=labels_dict_per_geom, 
    preds_dict_per_geom=preds_dict_per_geom, 
    freq_arrays_per_geom=freq_arrays_per_geom, 
    save_dir=f"out_files/single_mlp/touchstone_files"
)