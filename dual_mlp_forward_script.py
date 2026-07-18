import torch
from torch import nn
from load_set import create_param_dataloader, create_param_forward_dataloader
from dataset_splitting import split_dataset
from prediction.predictor import DeepBER_Param_Predictor
import numpy as np
import os


# ============================================= Initializing Dataset ============================================= #
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)

x_array = pred_arrays_dict["x_array"].astype(np.float32)
s_dict = pred_arrays_dict["s_dict"]
feature_columns = pred_arrays_dict["feature_columns"]

# Split the dataset into two parts: one for s-parameter (per-element) prediction and one for post-prediction physics enforcing
x_array_pred, pred_row_indices = split_dataset(x_array, sample_percentage=0.5, sampling_method="lhs", seed=seed)
s_dict_pred = {key: s_dict[key][pred_row_indices] for key in s_dict.keys()}

physics_row_indices = [idx for idx in range(x_array.shape[0]) if idx not in pred_row_indices]
x_array_physics = x_array[physics_row_indices]
s_dict_physics = {key: s_dict[key][physics_row_indices] for key in s_dict.keys()}

# Forward pass to get predictions of phyics-enforcement set
elements = list(key for key in s_dict.keys() if key != "all")
s_coarse_dict = {}
for element in elements:
    y_array = np.stack([s_dict_pred[element].real, s_dict_pred[element].imag], axis=1)
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    _, x_scale_params, y_scale_params = create_param_dataloader(
                    x_array_pred,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    forward_dataloader = create_param_forward_dataloader(
                    x_array_physics,
                    batch_size=128,
                    standard_scale=True,
                    x_scale_params=x_scale_params,
                    )
    
    predictor = DeepBER_Param_Predictor(
        input_size=len(feature_columns), 
        hidden=[48, 64, 96, 64, 48], 
        activation_fn=nn.GELU(), 
        output_size=out_size,
        dropout=0.02
        ).to(device)
    
    predictor.load_state_dict(torch.load(f"best_models/dual_mlp/best_model_{element}.pth", map_location=device))
    predictor.eval()

    all_preds = []
    with torch.no_grad():
        for inputs in forward_dataloader:
            inputs = inputs[0].to(device)
            outputs = predictor(inputs)

            all_preds.append(outputs.cpu().numpy())
    final_preds = np.concatenate(all_preds, axis=0)
    final_preds_unscaled = final_preds * y_scale_params[1] + y_scale_params[0]
    s_coarse_dict[element] = final_preds_unscaled

# Save s_coarse dictionary
pt_dir = "csv_files/s_params/pt"
os.makedirs(pt_dir, exist_ok=True)

s_coarse_dict_path = os.path.join(pt_dir, "s_coarse_dict.pt")
torch.save(s_coarse_dict, s_coarse_dict_path)

