import torch
import os
from load_set import create_param_prediction_arrays

torch.manual_seed(42)
csv_names = ["s_param_dataset_batch1.csv", "s_param_dataset_batch2.csv", "s_param_dataset_batch3.csv", "s_param_dataset_batch4.csv"]

x_array, s_dict, a_dict, b_dict, c_dict, d_dict, feature_columns = create_param_prediction_arrays(csv_names, sample_percentage=0.5, sampling_method="lhs", subfolder="s_params")

pred_arrays_dict = {
    "x_array": x_array,
    "s_dict": s_dict,
    "a_dict": a_dict,
    "b_dict": b_dict,
    "c_dict": c_dict,
    "d_dict": d_dict,
    "feature_columns": feature_columns
}

pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)


pred_arrays_path = os.path.join(pt_dir, "pred_arrays_dict_50perc.pt")
torch.save(pred_arrays_dict, pred_arrays_path)