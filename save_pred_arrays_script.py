import torch
import numpy as np
import os
from load_set import create_param_prediction_arrays


torch.manual_seed(42)
csv_names = [
    ["s_param_dataset_batch1.csv", "s_param_dataset_batch2.csv", "s_param_dataset_batch3.csv", "s_param_dataset_batch4.csv"],
]

test_names = ["param_prediction_test"]


pred_arrays_dict_30perc = create_param_prediction_arrays(csv_names, test_names, sample_percentage=0.3, sampling_method="lhs", subfolder="s_params")


pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)


pred_arrays_30perc_path = os.path.join(pt_dir, "pred_arrays_dict_30perc.pt")
torch.save(pred_arrays_dict_30perc, pred_arrays_30perc_path)