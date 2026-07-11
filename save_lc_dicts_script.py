import torch
import numpy as np
from prediction.s2abcd import s_dict2mat, s2lc_dict
from load_set import get_grouping
import os

# Load pred arrays
pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

freq = 5

# Filter the S-parameter dictionary to only include entries for the specified frequency
all_5GHz_indices = np.where(x_array[:, 7] == freq)[0]
s_dict_5GHz = {key: s_dict[key][all_5GHz_indices] for key in s_dict.keys()}
s_matrices_5GHz = s_dict2mat(s_dict_5GHz)

# length of the transmission lines for the specified frequency
lengths = x_array[all_5GHz_indices, 2] # !!!!!!!!

# Compute L and C matrices and output them as dictionaries
l_dict, c_dict = s2lc_dict(s_matrices_5GHz, freq, lengths)

# Save the L and C dictionaries as .pt files
pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)

l_dict_path = os.path.join(pt_dir, "l_dict.pt")
c_dict_path = os.path.join(pt_dir, "c_dict.pt")
torch.save(l_dict, l_dict_path)
torch.save(c_dict, c_dict_path)
