import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.param_pred_optuna import run_optuna
from load_set import organize_dataset_for_pi_stcnn, create_param_dataloader
from prediction.predictor import PI_STCNN
from prediction.l_freq_loss import l_freq_loss
from prediction.test_predictor_config import test_predictor_configuration_pistcnn
import numpy as np
from export_files_for_transient import export_files_for_transient, convert_stcnn_outputs_to_dicts

# ============================================= Initializing Dataset ============================================= #
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

x_array, feature_columns, y_array = organize_dataset_for_pi_stcnn(x_array, s_dict, feature_columns)

dataloader, x_scale_params, y_scale_params = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=16,
                    seed=42,
                    standard_scale=(True, True),  # (scale_features, scale_labels)
                    split_method="lhs"
                    )

_, num_channels_times2, num_freqs = y_array.shape
predictor = PI_STCNN(
    input_size=len(feature_columns),
    mlp_hidden=[30, 30],
    mlp_activation_fn=nn.ELU(),
    tcnn_layer_params=[
        [30, 32, 1],  # [out_channels, kernel_size, stride]
        [30, 4, 2],
        [30, 4, 2],
        [30, 4, 4],
        [30, 2, 3]
    ],
    tcnn_activation_fn=nn.ELU(),
    output_size=num_channels_times2 // 2,
    num_ports=18,
    N=num_freqs,
    M=1.5,
    K=2
).to(device)

criterion = l_freq_loss()
learning_rate = 0.001
weight_decay = 0.0 # 5.976118759714283e-06
optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5) # ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

test_preds, test_labels = test_predictor_configuration_pistcnn(
    title=f"S-Parameters Prediction with PI-STCNN",
    device=device,
    model=predictor,
    dataloader=dataloader,
    learning_rate=learning_rate,
    batch_size=128,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=3000,
    L_f=1000,
    early_stopping=True,
    patience=200,
    y_scale_params=y_scale_params,
    training_curves=True,
    predicted_vs_actual=True,
    test_out_dir = f"out_files/pi_stcnn",
    close_figures=True,
    max_figures=2,
    max_time_hours=5.5
    )

labels_dict_list, preds_dict_list = convert_stcnn_outputs_to_dicts(test_targets=test_labels, test_preds=test_preds, num_ports=18)

num_geometries = test_preds.shape[0]
freq_array = np.linspace(0, 30, 601)
freq_arrays_per_geom = [freq_array for _ in range(num_geometries)]

export_files_for_transient(
    geometries=x_array,  # The unique geometries returned by organize_dataset_for_pi_stcnn
    feature_names=feature_columns,
    labels_dict_per_geom=labels_dict_list,
    preds_dict_per_geom=preds_dict_list,
    freq_arrays_per_geom=freq_arrays_per_geom,
    save_dir="out_files/pi_stcnn/touchstone_files"
)