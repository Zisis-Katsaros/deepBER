import torch
from torch import nn
from load_set import create_param_prediction_arrays, create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor
from complexNN import nn as cvnn
from prediction.test_predictor_config import test_predictor_configuration
from rmse import RMSELoss

# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

csv_names = [
    ["s_param_dataset_batch1.csv", "s_param_dataset_batch2.csv", "s_param_dataset_batch3.csv", "s_param_dataset_batch4.csv"],
]

test_names = ["param_prediction_test"]

test_info_dict = create_param_prediction_arrays(csv_names, test_names, sampling_method="lhs", 
                                                  subfolder="s_params")

batch_size_dict = {
        "param_prediction_test": 16,
    }

dataloader_dict = {}

for test_name, test_info in test_info_dict.items():
        x_array, s_dict, a_dict, b_dict, c_dict, d_dict, feature_columns = test_info
        batch_size = batch_size_dict[test_name]

        dataloader_dict[test_name] = create_param_dataloader(
            x_array,
            a_dict["A12"].real,
            batch_size=batch_size,
            seed=42,
            standard_scale=True,
            split_method="lhs"
        )

feature_columns = test_info_dict["param_prediction_test"][6]
A12_dataloader = dataloader_dict["param_prediction_test"]

predictor = DeepBER_Param_Predictor(
    input_size=len(feature_columns) - 1, 
    hidden=[64,256,128, 64], 
    activation_fn=nn.GELU(), 
    dropout=0.02
    )

learning_rate = 0.001
criterion = RMSELoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
test_predictor_configuration(
        title="DeepBER A12 Parameter Prediction",
        device=device,
        model=predictor,
        dataloader=A12_dataloader,
        learning_rate=learning_rate,
        batch_size=16,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=240,
        early_stopping=True,
        patience=10,
        training_curves=True,
        predicted_vs_actual=True
    )