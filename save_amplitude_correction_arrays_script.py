import torch
import os
from load_set import create_amplitude_prediction_arrays

torch.manual_seed(42)
csv_names = ["transient_dataset_batch1.csv", "transient_dataset_batch2.csv", "transient_dataset_batch3.csv", "transient_dataset_batch4.csv"]

x_array, y_array, feature_columns = create_amplitude_prediction_arrays(csv_names, sample_percentage=1.0, sampling_method="lhs")

amplitude_pred_arrays_dict = {
    "x_array": x_array,
    "y_array": y_array,
    "feature_columns": feature_columns
}

pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)

amplitude_pred_arrays_path = os.path.join(pt_dir, "amplitude_pred_arrays_dict.pt")
torch.save(amplitude_pred_arrays_dict, amplitude_pred_arrays_path)