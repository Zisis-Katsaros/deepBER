import torch
from load_set import create_param_prediction_arrays, create_param_dataloader

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)

csv_names = [
    ["s_param_dataset_batch1_S12.csv", "s_param_dataset_batch2_S12.csv", "s_param_dataset_batch3_S12.csv"],
]

target_columns = [
    ["S12_real", "S12_imag"]
]

test_names = ["S12"]

test_info_dict = create_param_prediction_arrays(csv_names, target_columns, test_names, sampling_method="lhs", 
                                                  subfolder="s_params")

batch_size_dict = {
        "S12": 32,
    }

dataloader_dict = {}

for test_name, test_info in test_info_dict.items():
        x_array, y_array, _, _, thresholds, _ = test_info
        batch_size = batch_size_dict[test_name]

        dataloader_dict[test_name] = create_param_dataloader(
            x_array,
            y_array,
            batch_size=batch_size,
            seed=42,
            standard_scale=True,
            split_method="lhs"
        )
        
