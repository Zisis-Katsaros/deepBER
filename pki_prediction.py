import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from load_set import create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor
from rmse import RMSELoss
from prediction.test_predictor_config import test_predictor_configuration
import numpy as np
from prediction.s2abcd import trans_param_dict2mat
from prediction.pki_calculation import get_pki_dict, calculate_s_coarse_matrices
from visualization import plot_pki_vs_act_freq
from prediction.predictor_loops import test_pred_loop

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

pki_pred_arrays_dict = torch.load("csv_files/s_params/pt/pki_pred_arrays_dict.pt", weights_only=False)

x_array = pki_pred_arrays_dict["x_array"].astype(np.float32)
r_dict = pki_pred_arrays_dict["r_dict"]
l_dict = pki_pred_arrays_dict["l_dict"]
c_dict = pki_pred_arrays_dict["c_dict"]
g_dict = pki_pred_arrays_dict["g_dict"]
feature_columns = pki_pred_arrays_dict["feature_columns"]

r_preds_dict = {}
r_labels_dict = {}
l_preds_dict = {}
l_labels_dict = {}
c_preds_dict = {}
c_labels_dict = {}
g_preds_dict = {}
g_labels_dict = {}

elements = [key for key in r_dict.keys()] + [key for key in l_dict.keys()] + [key for key in c_dict.keys()] + [key for key in g_dict.keys()]
"""
for element in elements:
    print(f"Training and testing for {element}\n")
    
    if element in r_dict:
        y_array = r_dict[element]
    elif element in l_dict:
        y_array = l_dict[element]
    elif element in c_dict:
        y_array = c_dict[element]
    elif element in g_dict:
        y_array = g_dict[element]

    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    test_data = dataloader[2]
    batch_list = [input_batch.cpu().numpy() for input_batch, _ in test_data]
    x_array_test = np.concatenate(batch_list, axis=0).astype(np.float32)
    x_array_test = x_array_test * x_scale_params[1] + x_scale_params[0]
    lengths = x_array_test[:, 2] * 1e-6
    
    predictor = DeepBER_Param_Predictor(
        input_size=len(feature_columns), 
        hidden=[20, 20], 
        activation_fn=nn.GELU(), 
        output_size=out_size,
        dropout=0.01
        ).to(device)
    
    criterion = RMSELoss() # nn.MSELoss()
    learning_rate = 9.291819656627535e-05
    weight_decay = 5.976118759714283e-06
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
  
    test_preds, test_targets = test_predictor_configuration(
    title=f"{element}",
    device=device,
    model=predictor,
    dataloader=dataloader,
    learning_rate=learning_rate,
    batch_size=128,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=1,
    early_stopping=True,
    patience=10,
    y_scale_params=y_scale_params,
    training_curves=True,
    predicted_vs_actual=True,
    test_out_dir = f"out_files/pki_prediction/dual_mlp/{element}",
    close_figures=True
    )
"""

# Test forward pass
for element in elements:
    if element in r_dict:
        y_array = r_dict[element]
    elif element in l_dict:
        y_array = l_dict[element]
    elif element in c_dict:
        y_array = c_dict[element]
    elif element in g_dict:
        y_array = g_dict[element]

    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, x_scale_params, y_scale_params = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    test_data = dataloader[2]
    batch_list = [input_batch.cpu().numpy() for input_batch, _ in test_data]
    x_array_test = np.concatenate(batch_list, axis=0).astype(np.float32)
    x_array_test = x_array_test * x_scale_params[1] + x_scale_params[0]
    lengths = x_array_test[:, 2] * 1e-6
    
    predictor = DeepBER_Param_Predictor(
        input_size=len(feature_columns), 
        hidden=[20, 20], 
        activation_fn=nn.GELU(), 
        output_size=out_size,
        dropout=0.01
        ).to(device)
    predictor.load_state_dict(torch.load(f"out_files/pki_prediction/dual_mlp/{element}/best_model.pth", map_location=device))
    criterion = RMSELoss() # nn.MSELoss()
    
    _, _, _, test_preds, test_targets, *_, = test_pred_loop(predictor, test_data, criterion, device)

    if y_scale_params is not None:
        test_preds = test_preds * y_scale_params[1] + y_scale_params[0]
        test_targets = test_targets * y_scale_params[1] + y_scale_params[0]

    if element in r_dict:
        r_preds_dict[element] = test_preds
        r_labels_dict[element] = test_targets
    if element in l_dict:
        l_preds_dict[element] = test_preds
        l_labels_dict[element] = test_targets
    if element in c_dict:
        c_preds_dict[element] = test_preds
        c_labels_dict[element] = test_targets
    if element in g_dict:
        g_preds_dict[element] = test_preds
        g_labels_dict[element] = test_targets

r_matrices_preds = trans_param_dict2mat(r_labels_dict).astype(np.float64)
l_matrices_preds = trans_param_dict2mat(l_labels_dict).astype(np.float64)
c_matrices_preds = trans_param_dict2mat(c_labels_dict).astype(np.float64)
g_matrices_preds = trans_param_dict2mat(g_labels_dict).astype(np.float64)

freqs_ghz = np.linspace(0, 30, 601).astype(np.float64)

s_coarse_matrices = calculate_s_coarse_matrices(l_matrices_preds, c_matrices_preds, freqs_ghz=freqs_ghz, lengths=lengths, z0=50.0, r_matrices=r_matrices_preds, g_matrices=g_matrices_preds)

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
s_dict = pred_arrays_dict["s_dict"]
x_full = pred_arrays_dict["x_array"].astype(np.float32)
feature_columns_full = pred_arrays_dict["feature_columns"]

full_feature_indices = [feature_columns_full.index(name) for name in feature_columns]
freq_index_full = feature_columns_full.index("frequency_ghz")

# Aligning s and pki
num_test_geoms = x_array_test.shape[0]

for port_i in range(18):
    for port_j in range(port_i, 18):
        
        for geom_idx in range(num_test_geoms):
            match_mask = np.ones(x_full.shape[0], dtype=bool)
            for test_idx, feature_name in enumerate(feature_columns):
                full_idx = feature_columns_full.index(feature_name)
                match_mask &= np.isclose(x_full[:, full_idx], x_array_test[geom_idx, test_idx], rtol=1e-6, atol=1e-6)

            matched_indices = np.where(match_mask)[0]
            if matched_indices.size == 0:
                continue

            freqs_for_geom = x_full[matched_indices, freq_index_full]
            sort_order = np.argsort(freqs_for_geom)

            sorted_indices = matched_indices[sort_order]
            sorted_freqs = freqs_for_geom[sort_order]
            num_freqs = len(sorted_freqs)

            dict_i, dict_j = (port_i, port_j) if port_i <= port_j else (port_j, port_i)
            key = f"S{dict_i + 1}{dict_j + 1}"

            s_true_element = np.squeeze(s_dict[key][sorted_indices])
            s_coarse_element = np.squeeze(s_coarse_matrices[geom_idx, :, port_i, port_j])

            plot_pki_vs_act_freq(s_true_element.real, s_coarse_element.real, sorted_freqs, f"S{port_i + 1}{port_j + 1} (Real)")
            plot_pki_vs_act_freq(s_true_element.imag, s_coarse_element.imag, sorted_freqs, f"S{port_i + 1}{port_j + 1} (Imaginary)")








