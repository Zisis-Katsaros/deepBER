import torch
from load_set import create_param_prediction_arrays, create_param_dataloader

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)

csv_names = [
    ["s_param_dataset_batch1.csv"],
]

test_names = ["param_prediction_test"]

test_info_dict = create_param_prediction_arrays(csv_names, test_names, sampling_method="lhs", 
                                                  subfolder="s_params")

batch_size_dict = {
        "param_prediction_test": 32,
    }

dataloader_dict = {}

for test_name, test_info in test_info_dict.items():
        x_array, s_dict, a_dict, b_dict, c_dict, d_dict, feature_columns = test_info
        batch_size = batch_size_dict[test_name]

        dataloader_dict[test_name] = create_param_dataloader(
            x_array,
            s_dict["S12"],
            batch_size=batch_size,
            seed=42,
            standard_scale=True,
            split_method="lhs"
        )
        
