import torch
import os
from load_set import create_param_prediction_arrays

torch.manual_seed(42)
index = "unshielded"
csv_names = [f"s_param_dataset_batch1_{index}.csv", f"s_param_dataset_batch2_{index}.csv", f"s_param_dataset_batch3_{index}.csv", f"s_param_dataset_batch4_{index}.csv"]

x_array, s_dict, feature_columns = create_param_prediction_arrays(csv_names, sample_percentage=1, sampling_method="lhs", subfolder=f"s_params/{index}")

pred_arrays_dict = {
    "x_array": x_array,
    "s_dict": s_dict,
    # "a_dict": a_dict,
    # "b_dict": b_dict,
    # "c_dict": c_dict,
    # "d_dict": d_dict,
    "feature_columns": feature_columns
}

pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)


pred_arrays_path = os.path.join(pt_dir, f"pred_arrays_dict_{index}.pt")
torch.save(pred_arrays_dict, pred_arrays_path)