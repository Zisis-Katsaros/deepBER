import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.pi_stcnn_optuna import run_pi_stcnn_optuna
from load_set import organize_dataset_for_pi_stcnn, create_param_dataloader
from prediction.predictor import PI_STCNN
from prediction.l_freq_loss import l_freq_loss
from prediction.test_predictor_config import test_predictor_configuration_pistcnn
import numpy as np
from export_files_for_transient import export_files_for_transient, convert_stcnn_outputs_to_dicts
from dataset_splitting import split_dataset
from prediction.param_pred_optuna_helpers import eval_pistcnn_study


# ============================================= Initializing Dataset ============================================= #
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

pki = False
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

# x_array, s_dict = cut_dataset_at_specified_freq(x_array, s_dict, feature_columns, cutoff_freq_ghz=20.0)

if pki:
    s_coarse_dict = torch.load("csv_files/s_params/pt/s_coarse_dict.pt", weights_only=False)
    _, pred_row_indices = split_dataset(x_array, sample_percentage=0.5, sampling_method="lhs", seed=seed)
    physics_row_indices = [idx for idx in range(x_array.shape[0]) if idx not in pred_row_indices]
    x_array_physics = x_array[physics_row_indices]
    s_dict_physics = {key: s_dict[key][physics_row_indices] for key in s_dict.keys()}

    x_array, feature_columns, y_array, pki_array = organize_dataset_for_pi_stcnn(x_array_physics, s_dict_physics, feature_columns, pki_dict=s_coarse_dict)
else:
    x_array, feature_columns, y_array = organize_dataset_for_pi_stcnn(x_array, s_dict, feature_columns)

"""
db_path = "out_files/pi_stcnn/pi_stcnn_study3.db"
storage_url = f"sqlite:///{db_path}"
# eval_pistcnn_study(storage_url)
run_pi_stcnn_optuna(x_array, y_array, feature_columns, n_trials=400, n_epochs=850, storage=storage_url)

"""
dataloader, x_scale_params, y_scale_params, y_weights, *_ = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=16,
                    seed=42,
                    standard_scale=(True, False),  # (scale_features, scale_labels)
                    split_method="lhs",
                    pki_array=pki_array if pki else None,
                    weight_type="balanced"
                    )

_, num_channels_times2, num_freqs = y_array.shape
predictor = PI_STCNN(
    input_size=len(feature_columns),
    mlp_hidden=[64, 64, 64, 64],
    mlp_activation_fn=nn.ELU(),
    mlp_dropout=0.0,
    tcnn_layer_params=[
        [196, 0, 0],  # [out_channels, kernel_size, stride]
        [196, 4, 2],
        [196, 3, 3],
        [196, 8, 4],
        [196, 8, 2],
        [196, 2, 2]
    ],
    tcnn_activation_fn=nn.ELU(),
    output_size=num_channels_times2 // 2,
    num_ports=18,
    N=num_freqs,
    M=1.5,
    K=2,
    varience_min=0.05,
    layer_norm=True,
).to(device)

criterion = l_freq_loss(weight=y_weights).to(device)
learning_rate = 0.001
weight_decay = 0.0 # 5.1635e-05
optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100) # torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5) 

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
    phase1_patience=50,
    early_stopping=True,
    patience=300,
    y_scale_params=y_scale_params,
    training_curves=True,
    predicted_vs_actual=True,
    test_out_dir = f"out_files/pi_stcnn",
    close_figures=True,
    max_figures=3,
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
# """