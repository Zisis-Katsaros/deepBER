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
from prediction.s2abcd import s_param_imag_part_hilbert_construction
from export_files_for_transient import export_files_for_transient, convert_stcnn_outputs_to_dicts

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

s_non_causal_dict = {}
processed_elements = [key for idx, key in enumerate(s_dict.keys()) if idx < -1 and key != "all"]
elements = list(key for key in s_dict.keys() if key != "all") 
for element in elements:   
    if element in processed_elements:
        continue
    print(f"Training and testing for {element}\n")

    y_array = s_dict[element].real
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params, _ = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    predictor = DeepBER_Param_Predictor(
        input_size=x_array.shape[1],
        hidden=[128, 128, 128, 128],
        activation_fn=nn.GELU(),
        output_size=out_size,
        dropout=0.04
        ).to(device)
    
    criterion = RMSELoss()
    learning_rate = 0.000446
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
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
    test_out_dir = f"out_files/single_mlp",
    close_figures=True
    )

# Extrapolation, Causality Enforcement and Exporting for Transient Simulation
for element in elements:
    y_array = s_dict[element].real
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params, _ = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    test_data, x_extrap, x_test_original = add_samples_for_extrapolation(
        test_data=dataloader[2],
        M=1.5,
        feature_columns=feature_columns,
        x_scale_params=x_scale_params,
        n_non_unique_feats=7
    )

    predictor = DeepBER_Param_Predictor(
        input_size=x_array.shape[1],
        hidden=[128, 128, 128, 128],
        activation_fn=nn.GELU(),
        output_size=out_size,
        dropout=0.04
        ).to(device)
    predictor.load_state_dict(torch.load(f"out_files/single_mlp/weights/best_model_{element}.pth", map_location=device))
    
    # Per element act vs pred plots on frequency domain
    single_geometry_test(
        title=f"{element}",
        device=device,
        model=predictor,
        test_data = dataloader[2],
        x_scale_params=x_scale_params,
        y_scale_params=y_scale_params,
        max_geoms=3,
        visualization=True,
        save_dir = f"out_files/single_mlp/{element}",
        close_figures=True
    )

    all_preds = []
    # Forward pass with extrapolated test data
    predictor.eval()
    with torch.no_grad(): # disable gradient calculation
        for inputs in test_data:
            inputs = inputs[0].to(device)
            outputs = predictor(inputs)
            
            all_preds.append(outputs.cpu().numpy())
    final_preds = np.concatenate(all_preds, axis=0)
    final_preds_unscaled = final_preds * y_scale_params[1] + y_scale_params[0]
    s_non_causal_dict[element] = final_preds_unscaled

unique_geoms, new_feature_columns, s_non_causal = organize_dataset_for_pi_stcnn(x_extrap, s_non_causal_dict, feature_columns, only_real=True)

K=2
s_causal = s_param_imag_part_hilbert_construction(s_non_causal, num_og_freq=601, K=K)

if K>1:
    s_causal = s_causal[:, :, ::K]  # Downsample to match original frequency points

test_preds = np.concatenate([s_causal.real, s_causal.imag], axis=1).astype(np.float32)

# Convert labels to 3D array
test_indices = []
for test_row in x_test_original:
    test_row = test_row * x_scale_params[1] + x_scale_params[0]  # Unscale the test row
    match_idx = np.where(np.all(np.isclose(x_array, test_row, atol=1e-6), axis=1))[0]
    if len(match_idx) > 0:
        test_indices.append(match_idx[0]) # grab first match
test_mask = np.array(test_indices)
s_dict_test = {key: s_dict[key][test_mask] for key in s_dict.keys() if key != "all"}
_, _, test_targets = organize_dataset_for_pi_stcnn(x_array=x_test_original, s_dict=s_dict_test, feature_columns=feature_columns, freq_round_decimals=6)

# Convert to per geometry dictionary
labels_dict_per_geom, preds_dict_per_geom = convert_stcnn_outputs_to_dicts(test_targets=test_targets, test_preds=test_preds, num_ports=18)

freq_array = np.linspace(0, 30, 601)
# Duplicate the frequency array for every unique geometry in the batch
freq_arrays_per_geom = [freq_array for _ in range(len(unique_geoms))]

export_files_for_transient(
    geometries=unique_geoms, 
    feature_names=new_feature_columns, 
    labels_dict_per_geom=labels_dict_per_geom, 
    preds_dict_per_geom=preds_dict_per_geom, 
    freq_arrays_per_geom=freq_arrays_per_geom, 
    save_dir=f"out_files/single_mlp/touchstone_files"
)