import torch
import numpy as np
from dataset_manipulation import mock_pki
from visualization import plot_pki_vs_act_freq
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)

x_array = pred_arrays_dict["param_prediction_test"][0]
s_dict = pred_arrays_dict["param_prediction_test"][1]
feature_columns = pred_arrays_dict["param_prediction_test"][6]

sample_ids = np.array([row[0] for row in x_array])
freqs = np.array([float(row[8]) for row in x_array])

# Initialize s_mock_dict
s_mock_dict = {key: np.zeros_like(s_param, dtype=complex) for key, s_param in s_dict.items()}

unique_samples = np.unique(sample_ids)
for sample in unique_samples:
    idx = np.where(sample_ids == sample)[0]
    sample_freqs = freqs[idx]
    
    # Sort freqs as interp1d used in mock_pki requires sorted argument
    sort_mask = np.argsort(sample_freqs)
    sample_freqs_sorted = sample_freqs[sort_mask]
    
    for key in s_dict:
        if key == "all":
            continue
        # Isolate the S-parameter array for this sample
        sample_s_param = np.array(s_dict[key])[idx]
        sample_s_param_sorted = sample_s_param[sort_mask]
        
        # Apply the mock PKI function
        mocked_s_param_sorted = mock_pki(sample_freqs_sorted, sample_s_param_sorted)
        
        # Un-sort to restore the original row ordering and store back into s_mock_dict
        original_order_mask = np.argsort(sort_mask)
        s_mock_dict[key][idx] = mocked_s_param_sorted[original_order_mask]

        # Plot PKI vs Actual for visualization
        # plot_pki_vs_act_freq(sample_s_param_sorted.real, mocked_s_param_sorted.real, sample_freqs_sorted)

pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)
s_mock_dict_30perc_path = os.path.join(pt_dir, "s_mock_dict.pt")
torch.save(s_mock_dict, s_mock_dict_30perc_path)
