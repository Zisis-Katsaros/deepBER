import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from load_set import create_param_dataloader, create_amplitude_correction_dataloader, organize_dataset_for_pi_stcnn
from prediction.predictor import DeepBER_Param_Predictor
from prediction.test_predictor_config import test_predictor_configuration
from rmse import RMSELoss
import numpy as np
from prediction.amplitude_correction_optuna import run_amp_corr_optuna
from export_files_for_transient import export_amplitude_correction
from scipy.spatial import cKDTree

# ============================================= Initializing Dataset ============================================= #
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load amplitude correction arrays
amplitude_pred_arrays_dict = torch.load("csv_files/s_params/pt/amplitude_pred_arrays_dict.pt", weights_only=False)
x_array_amplitude = amplitude_pred_arrays_dict["x_array"].astype(np.float32)
y_array_amplitude = amplitude_pred_arrays_dict["y_array"]
feature_columns_amplitude = amplitude_pred_arrays_dict["feature_columns"]

# """
# Load s-parameter prediction arrays
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]
x_array, feature_columns, y_array = organize_dataset_for_pi_stcnn(x_array, s_dict, feature_columns)

# Realign the amplitude dataset to match the order of x_array

_, aligned_indices = cKDTree(x_array_amplitude).query(x_array)
x_array_amplitude = x_array_amplitude[aligned_indices]
y_array_amplitude = y_array_amplitude[aligned_indices]

"""
db_path = "out_files/amplitude_prediction/amp_corr_study.db"
storage_url = f"sqlite:///{db_path}"
run_amp_corr_optuna(x_array_amplitude, y_array_amplitude, feature_columns_amplitude, n_trials=600, n_epochs=80, storage=storage_url)

"""

# Create s-parameter dataloader
dataloader, x_scale_params, y_scale_params, _, split_idx = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=16,
                    seed=42,
                    standard_scale=(True, False),  # (scale_features, scale_labels)
                    split_method="lhs",
                    )

# Create amplitude correction dataloader
amplitude_correction_dataloader, x_scale_params_amp, y_scale_params_amp, *_ = create_param_dataloader(
                    x_array_amplitude,
                    y_array_amplitude,
                    batch_size=16,
                    seed=42,
                    standard_scale=(True, False),  # (scale_features, scale_labels)
                    split_method="custom",
                    split_idx=split_idx,
                    )

# Initialize predictor DNN
predictor = DeepBER_Param_Predictor(
    input_size=len(feature_columns_amplitude),
    hidden=[64, 128, 256, 128, 64],
    output_size=1,
    activation_fn=torch.nn.ELU(),
)

criterion = RMSELoss()
learning_rate = 0.001
weight_decay = 1.04019e-06
optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

test_inputs, test_preds, test_targets = test_predictor_configuration(
    title="DeepBER Amplitude Prediction",
    device=device,
    model=predictor,
    dataloader=amplitude_correction_dataloader,
    learning_rate=learning_rate,
    batch_size=16,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=1000,
    early_stopping = True,
    patience=50,
    x_scale_params=x_scale_params_amp,
    y_scale_params=y_scale_params_amp,
    training_curves=True,
    predicted_vs_actual=True,
    test_out_dir = "out_files/amplitude_prediction",
    close_figures=False,
)


export_amplitude_correction(test_inputs, test_preds, test_targets, feature_names=feature_columns_amplitude, save_dir="out_files/amplitude_prediction/export4transient")
# """


